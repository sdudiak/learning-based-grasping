from itertools import product
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


NEIGHBORS = np.array(list(product((-1, 0, 1), repeat=3)))


class TargetMap:
    def __init__(self, resolution, angle_deg, minimum_overlap):
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Voxel resolution must be positive and finite')
        if not 0 < angle_deg <= 180 or not 0 < minimum_overlap <= 1:
            raise ValueError('Invalid viewpoint angle or overlap threshold')
        self.resolution = resolution
        self.direction_limit = np.cos(np.deg2rad(angle_deg))
        self.minimum_overlap = minimum_overlap
        self.clear()

    def clear(self):
        self.anchor = None
        self.directions = []
        self.views = []
        self.voxels = {}
        self.neighborhood = set()

    def add(self, points, stamp, pose):
        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 32:
            return 'Too few finite target points'
        if self.views and stamp <= self.views[-1][0]:
            return 'Observation is older than the last accepted view'
        anchor = points.mean(axis=0) if self.anchor is None else self.anchor
        direction = np.asarray(pose[:3]) - anchor
        distance = np.linalg.norm(direction)
        if not np.isfinite(distance) or distance < 1e-6:
            return 'Invalid camera position relative to target'
        direction /= distance
        if any(direction @ previous > self.direction_limit for previous in self.directions):
            return 'Viewpoint already captured'

        keys, inverse, counts = np.unique(
            np.floor(points / self.resolution).astype(np.int64), axis=0,
            return_inverse=True, return_counts=True)
        voxel_keys = list(map(tuple, keys))
        if self.voxels:
            overlap = sum(key in self.neighborhood for key in voxel_keys) / len(voxel_keys)
            if overlap < self.minimum_overlap:
                return 'Target surface overlap is too low'

        sums = np.zeros((len(keys), 3))
        np.add.at(sums, inverse, points)
        view_id = len(self.views)
        for key, centroid in zip(voxel_keys, sums / counts[:, None]):
            self.voxels.setdefault(key, {})[view_id] = centroid
        neighbors = (keys[:, None, :] + NEIGHBORS).reshape(-1, 3)
        self.neighborhood.update(map(tuple, neighbors))
        self.anchor = anchor
        self.directions.append(direction)
        self.views.append((stamp, pose, points))
        return None

    def points(self):
        return np.asarray(
            [np.mean(list(samples.values()), axis=0) for samples in self.voxels.values()],
            dtype=np.float32).reshape(-1, 3)

    def save(self, path, frame_id, camera_frame, selection_id):
        points = self.points()
        if len(points) < 32 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError('The map needs at least 32 voxels spanning a surface')

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamps, poses, views = zip(*self.views)
        contributors = list(self.voxels.values())
        metadata = dict(frame_id=frame_id, units='meters', camera_frame=camera_frame,
                        selection_id=selection_id, voxel_resolution_m=self.resolution)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(
                    stream, points=points, metadata=json.dumps(metadata),
                    view_points=np.concatenate(views),
                    view_offsets=np.r_[0, np.cumsum([len(view) for view in views])],
                    view_stamps_ns=np.asarray(stamps, dtype=np.int64),
                    camera_poses_xyz_xyzw=np.asarray(poses),
                    voxel_view_ids=np.asarray([i for views in contributors for i in views]),
                    voxel_view_offsets=np.r_[0, np.cumsum([len(views) for views in contributors])],
                )
                stream.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path
