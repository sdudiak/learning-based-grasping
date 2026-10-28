import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import GroundPlane
from pxr import Gf, Usd, UsdGeom, UsdLux

from isaac_manipulation.object_spawner.spawning import SettlingTracker, drop_object, prepare_object


class Scene:
    def __init__(self, rng):
        self.rng = rng
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30, device='cpu')
        self.world.get_physics_context().enable_stablization(True)
        material = PhysicsMaterial('/World/GroundMaterial', static_friction=0.7, dynamic_friction=0.6, restitution=0)
        self.world.scene.add(GroundPlane('/World/Ground', size=200, color=np.array([0.35, 0.35, 0.35]), physics_material=material))
        self.stage = self.world.stage
        self.bodies = []
        self.objects = []
        self.camera = UsdGeom.Camera.Define(self.stage, '/World/Camera')
        self.camera.CreateFocalLengthAttr(24.0)
        self.camera.CreateHorizontalApertureAttr(36.0)
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        self.lighting = {'position': [0.6, -0.6, 1.5], 'intensity': 10000.0,
                         'temperature_kelvin': 6500.0, 'radius': 0.5, 'normalize': True,
                         'fill_intensity': 250.0}
        light = UsdLux.SphereLight.Define(self.stage, '/World/Light')
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*self.lighting['position']))
        light.CreateRadiusAttr(self.lighting['radius'])
        light.CreateNormalizeAttr(True)
        light.CreateIntensityAttr(self.lighting['intensity'])
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(self.lighting['temperature_kelvin'])
        fill = UsdLux.DomeLight.Define(self.stage, '/World/Fill')
        fill.CreateIntensityAttr(self.lighting['fill_intensity'])

    def spawn(self, assets, count, spread=0.03):
        self.world.stop()
        for body in self.bodies:
            self.world.scene.remove_object(body.name)
        self.bodies = []
        self.objects = []
        selected = self.rng.choice(len(assets), size=count, replace=count > len(assets))
        prepared = []
        for i, index in enumerate(selected):
            asset = assets[int(index)]
            path = f'/World/Objects/instance_{i:04d}'
            body, pose, metadata = prepare_object(self.stage, asset, path, i, self.rng, spread)
            prepared.append(pose)
            self.world.scene.add(body)
            self.bodies.append(body)
            self.objects.append(metadata)
        self.world.reset()
        for i, (body, pose) in enumerate(zip(self.bodies, prepared)):
            drop_object(self.stage, body, pose, self.objects[:i], self.objects[i])
            for _ in range(30):
                self.world.step(render=False)
        return self.objects

    def settle(self, timeout, linear_threshold=0.01, angular_threshold=0.15):
        tracker = SettlingTracker(1 / 120, linear_threshold=linear_threshold, angular_threshold=angular_threshold)
        for step in range(int(timeout * 120)):
            self.world.step(render=False)
            result = tracker.sample(self.bodies)
            if result and (not result['settled'] or step >= 120):
                self.world.pause()
                if result['settled']:
                    for body, obj in zip(self.bodies, self.objects):
                        position, orientation = body.get_world_pose()
                        obj.update(position=position.tolist(), orientation_wxyz=orientation.tolist())
                return {**result, 'seconds': (step + 1) / 120}
        self.world.pause()
        return {'settled': False, 'seconds': timeout, 'reason': 'settling timeout'}

    def randomize_view(self, resolution):
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        ranges = [bounds.ComputeWorldBound(self.stage.GetPrimAtPath(o['prim_path'])).ComputeAlignedRange() for o in self.objects]
        low = np.min([r.GetMin() for r in ranges], axis=0)
        high = np.max([r.GetMax() for r in ranges], axis=0)
        target = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) / 2, 0.08)
        target += self.rng.uniform(-0.12, 0.12, size=3) * radius
        aspect = resolution[0] / resolution[1]
        focal_length = float(self.rng.uniform(16, 40))
        azimuth = self.rng.uniform(0, 2 * np.pi)
        elevation = self.rng.uniform(np.deg2rad(10), np.deg2rad(88))
        direction = np.array([np.cos(azimuth) * np.cos(elevation),
                              np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
        roll = float(self.rng.uniform(-35, 35))
        right = np.cross([0, 0, 1], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        angle = np.deg2rad(roll)
        camera_right = np.cos(angle) * right + np.sin(angle) * up
        camera_up = -np.sin(angle) * right + np.cos(angle) * up
        corners = np.array([[x, y, z] for x in (low[0], high[0])
                            for y in (low[1], high[1]) for z in (low[2], high[2])]) - target
        depth = corners @ direction
        horizontal_fit = depth + np.abs(corners @ camera_right) / (36 / (2 * focal_length))
        vertical_fit = depth + np.abs(corners @ camera_up) / (36 / (2 * focal_length * aspect))
        distance = max(float(horizontal_fit.max()), float(vertical_fit.max()), 0.1) * self.rng.uniform(1.05, 1.15)
        position = target + distance * direction
        transform = UsdGeom.Xformable(self.camera)
        transform.ClearXformOpOrder()
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), roll)) * matrix
        transform.AddTransformOp().Set(matrix)
        self.camera.CreateFocalLengthAttr(focal_length)
        self.camera.CreateVerticalApertureAttr(36.0 / aspect)
        return {'camera': {'position': position.tolist(), 'target': target.tolist(),
                           'camera_to_world': [list(row) for row in matrix],
                           'azimuth_deg': float(np.rad2deg(azimuth)), 'elevation_deg': float(np.rad2deg(elevation)),
                           'roll_deg': roll, 'distance_m': distance, 'focal_length_mm': focal_length, 'horizontal_aperture_mm': 36,
                           'vertical_aperture_mm': 36 / aspect, 'resolution': list(resolution)},
                'light': dict(self.lighting)}
