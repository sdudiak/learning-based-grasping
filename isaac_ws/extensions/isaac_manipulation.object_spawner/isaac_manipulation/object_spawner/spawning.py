from collections import deque
from pathlib import Path

import numpy as np
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


ROOT_PATH = '/World/SpawnedObjects'
OWNER = 'isaac_manipulation.object_spawner'


def find_assets(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'Asset directory does not exist: {directory}')
    assets = sorted(path / f'{path.name}.usd' for path in directory.iterdir()
                    if path.is_dir() and (path / f'{path.name}.usd').is_file())
    if not assets:
        raise ValueError(f'No converted GSO assets in {directory}')
    return assets


def bounds(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    low, high = np.array(extent.GetMin()), np.array(extent.GetMax())
    if not np.isfinite([low, high]).all() or np.max(high - low) <= 0:
        raise ValueError(f'Invalid bounds: {prim.GetPath()}')
    return low, high


def owned_root(stage):
    prim = stage.GetPrimAtPath(ROOT_PATH)
    if prim and prim.GetCustomDataByKey('objectSpawnerOwner') != OWNER:
        raise ValueError(f'{ROOT_PATH} exists and is not owned by the object spawner')
    return prim


def clear_objects(stage):
    if owned_root(stage):
        stage.RemovePrim(ROOT_PATH)


def prepare_object(stage, asset, path, index, rng, spread, surface_bounds=None, body_factory=SingleRigidPrim):
    prim = add_reference_to_stage(str(asset), path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    physics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physics.CreateSolverPositionIterationCountAttr(16)
    physics.CreateSolverVelocityIterationCountAttr(4)
    physics.CreateSleepThresholdAttr(0.001)
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(child)
            collision.CreateContactOffsetAttr(0.002)
            collision.CreateRestOffsetAttr(0.0)
    bounds(prim)
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    body = body_factory(path, name=f'instance_{index:04d}', position=np.zeros(3), orientation=quaternion)
    try:
        low, high = bounds(prim)
        if surface_bounds is not None:
            surface_low, surface_high = surface_bounds
            available = (surface_high[:2] - surface_low[:2] - (high[:2] - low[:2])) / 2
            if np.any(available < 0):
                raise ValueError(f'Object does not fit on the target surface: {asset}')
    except Exception:
        body._prim_view.destroy()
        stage.RemovePrim(path)
        raise
    center = (low + high) / 2
    target = np.zeros(2)
    half_width = np.full(2, spread)
    if surface_bounds is not None:
        surface_low, surface_high = surface_bounds
        target = (surface_low[:2] + surface_high[:2]) / 2
        half_width = np.minimum(half_width, available)
    xy = target + rng.uniform(-half_width, half_width, size=2) - center[:2]
    parked = np.array([10.0 + index, 0.0, 1.0])
    body.set_world_pose(parked, quaternion)
    body.set_default_state(parked, quaternion)
    body.set_visibility(False)
    add_update_semantics(prim, 'object', type_label='class')
    metadata = {'id': index + 1, 'prim_path': path, 'asset': str(asset), 'asset_name': asset.stem,
                'spawn_orientation_wxyz': quaternion.tolist(), 'drop_clearance_m': 0.015}
    return body, (xy, float(low[2]), quaternion), metadata


class SpawnBody(SingleXFormPrim):
    def __init__(self, *args, **kwargs):
        self.physics_view = None
        self.index = None
        super().__init__(*args, **kwargs)

    def destroy(self):
        self.physics_view = None
        self._prim_view.destroy()

    def get_world_pose(self):
        if self.physics_view is None:
            return super().get_world_pose()
        pose = self.physics_view.get_transforms()[self.index]
        return pose[:3].copy(), pose[[6, 3, 4, 5]].copy()

    def set_world_pose(self, position=None, orientation=None):
        if self.physics_view is None:
            return super().set_world_pose(position, orientation)
        poses = self.physics_view.get_transforms()
        if position is not None:
            poses[self.index, :3] = position
        if orientation is not None:
            poses[self.index, 3:] = np.asarray(orientation)[[1, 2, 3, 0]]
        self.physics_view.set_transforms(poses, np.array([self.index], dtype=np.uint32))

    def get_linear_velocity(self):
        return self.physics_view.get_velocities()[self.index, :3].copy()

    def get_angular_velocity(self):
        return self.physics_view.get_velocities()[self.index, 3:].copy()

    def set_linear_velocity(self, velocity):
        self._set_velocity(velocity, slice(0, 3))

    def set_angular_velocity(self, velocity):
        self._set_velocity(velocity, slice(3, 6))

    def _set_velocity(self, velocity, component):
        values = self.physics_view.get_velocities()
        values[self.index, component] = velocity
        self.physics_view.set_velocities(values, np.array([self.index], dtype=np.uint32))


def drop_object(stage, body, prepared, previous_objects, metadata, surface_height=0.0):
    xy, bottom, quaternion = prepared
    pile_top = max([surface_height] + [bounds(stage.GetPrimAtPath(obj['prim_path']))[1][2]
                                       for obj in previous_objects])
    position = np.array([xy[0], xy[1], pile_top + 0.015 - bottom])
    body.set_world_pose(position, quaternion)
    body.set_linear_velocity(np.zeros(3))
    body.set_angular_velocity(np.zeros(3))
    body.set_visibility(True)
    metadata['spawn_position'] = position.tolist()


class SettlingTracker:
    def __init__(self, physics_dt, minimum_height=-0.5, linear_threshold=0.01, angular_threshold=0.15):
        self.minimum_height = minimum_height
        self.limits = np.array([linear_threshold, angular_threshold])
        window = max(2, round(0.5 / physics_dt))
        self.speeds = deque(maxlen=window)
        self.poses = deque(maxlen=window)

    def sample(self, bodies):
        current = [body.get_world_pose() for body in bodies]
        positions = np.array([pose[0] for pose in current])
        orientations = np.array([pose[1] for pose in current])
        if not np.isfinite(positions).all() or not np.isfinite(orientations).all() or np.any(positions[:, 2] < self.minimum_height):
            return {'settled': False, 'reason': 'invalid object pose'}
        self.poses.append((positions, orientations))
        speeds = [[np.linalg.norm(body.get_linear_velocity()), np.linalg.norm(body.get_angular_velocity())]
                  for body in bodies]
        if not np.isfinite(speeds).all():
            return {'settled': False, 'reason': 'invalid object velocity'}
        self.speeds.append(speeds)
        if len(self.speeds) < self.speeds.maxlen:
            return None
        # Average a half-second window to tolerate contact-solver jitter.
        stable = np.all(np.mean(self.speeds, axis=0) < self.limits) and np.all(np.max(self.speeds, axis=0) < 3 * self.limits)
        position_window = np.array([pose[0] for pose in self.poses])
        orientation_window = np.array([pose[1] for pose in self.poses])
        drift = np.linalg.norm(np.ptp(position_window, axis=0), axis=1)
        dots = np.abs(np.sum(orientation_window * orientation_window[0], axis=2))
        rotation = 2 * np.arccos(np.clip(dots, 0, 1))
        pose_stable = np.all(drift < 0.002) and np.all(rotation < np.deg2rad(1))
        if stable or pose_stable:
            return {'settled': True, 'criterion': 'velocity' if stable else 'pose stability'}
        return None
