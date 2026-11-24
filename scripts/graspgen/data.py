import json
from pathlib import Path

import numpy as np


def validate_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected an (N, 3) point cloud, got {points.shape}')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 32:
        raise ValueError('At least 32 finite points are required')
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError('Point cloud must span at least a surface')
    return np.ascontiguousarray(points)


def load_input(path, scale=1.0, num_points=2048, seed=42):
    path = Path(path)
    metadata = {'source': str(path.resolve()), 'units': 'meters', 'frame_id': 'unspecified'}
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            points = data['points']
            if 'metadata' in data:
                metadata.update(json.loads(str(data['metadata'])))
    elif path.suffix == '.npy':
        points = np.load(path, allow_pickle=False)
    elif path.suffix == '.json':
        data = json.loads(path.read_text())
        points = data['pc']
        metadata.update(data.get('metadata', {}))
    elif path.suffix.lower() in ('.obj', '.stl', '.ply'):
        import trimesh
        mesh = trimesh.load(path, force='mesh')
        mesh.apply_scale(scale)
        points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
        metadata.update(frame_id='object', mesh_scale=scale)
    else:
        raise ValueError('Supported inputs: .npz, .npy, upstream .json, .obj, .stl, mesh .ply')
    if metadata.get('units') != 'meters':
        raise ValueError('Input coordinates must be in meters')
    points = validate_points(points)
    metadata['original_point_count'] = len(points)
    if len(points) > num_points:
        points = points[np.random.default_rng(seed).choice(len(points), num_points, replace=False)]
    metadata['input_point_count'] = len(points)
    metadata['bounds_m'] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    return points, metadata


def validate_predictions(grasps, scores):
    grasps = np.asarray(grasps, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or grasps.shape != (len(scores), 4, 4):
        raise ValueError(f'Malformed predictions: {grasps.shape}, {scores.shape}')
    if not len(scores):
        raise ValueError('No grasp candidates returned')
    if not np.isfinite(grasps).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite grasp predictions')
    if not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError('Confidence scores must lie in [0, 1]')
    rotations = grasps[:, :3, :3]
    if not np.allclose(grasps[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid homogeneous transforms')
    if not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-3):
        raise ValueError('Grasp rotations are not orthonormal')
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError('Grasp rotations must be right handed')
    return grasps, scores


def save_cloud(path, points, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, points=validate_points(points), metadata=json.dumps(metadata))


def top_down_mask(grasps, angle_deg):
    return -grasps[:, 2, 2] >= np.cos(np.deg2rad(angle_deg))
