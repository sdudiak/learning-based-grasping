import hashlib
import json
from pathlib import Path

import numpy as np


def load_instances(path, shape, predictions=False):
    with np.load(path, allow_pickle=False) as data:
        masks = data['masks']
        if masks.dtype != np.bool_:
            raise ValueError(f'Masks must be boolean: {path}')
        if masks.ndim != 3 or masks.shape[1:] != shape:
            raise ValueError(f'Invalid mask dimensions: {path}')
        if not predictions and len(masks) and not masks.any(axis=(1, 2)).all():
            raise ValueError(f'Ground-truth masks must not be empty: {path}')
        if 'boxes' in data:
            boxes = np.asarray(data['boxes'], dtype=np.float64)
        else:
            boxes = []
            for mask in masks:
                y, x = np.nonzero(mask)
                if not x.size:
                    raise ValueError(f'Cannot derive a box from an empty mask: {path}')
                boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
            boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if boxes.shape != (len(masks), 4) or masks.shape != (len(boxes), *shape):
            raise ValueError(f'Invalid boxes/masks dimensions: {path}')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError(f'Invalid bounding boxes: {path}')
        result = {'boxes': boxes, 'masks': masks}
        if predictions:
            scores = np.asarray(data['scores'], dtype=np.float64)
            if scores.shape != (len(boxes),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f'Invalid confidence scores: {path}')
            result['scores'] = scores
    return result


def load_prediction_run(predictions, dataset_override=None):
    predictions = Path(predictions).expanduser().resolve()
    if not predictions.is_dir():
        raise ValueError(f'Prediction directory does not exist: {predictions}')
    manifest_path = predictions.parent / 'manifest.json'
    run = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest_path.is_file() and run.get('status') != 'complete':
        raise ValueError('Prediction run must complete successfully before evaluation')
    dataset_path = dataset_override or run.get('dataset')
    if not dataset_path:
        raise ValueError('No dataset associated with these predictions; provide --dataset')
    dataset = Path(dataset_path).expanduser().resolve()
    raw = (dataset / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest.get('status') != 'complete':
        raise ValueError('Dataset generation or preparation must complete before evaluation')
    fingerprint = run.get('dataset_manifest_sha256')
    if fingerprint is not None and hashlib.sha256(raw).hexdigest() != fingerprint:
        raise ValueError('Dataset manifest differs from the dataset used for prediction')
    frames = manifest['frames']
    by_id = {frame['id']: frame for frame in frames}
    if not frames or len(by_id) != len(frames):
        raise ValueError('Dataset must contain frames with unique IDs')
    selected = run.get('frame_ids', list(by_id))
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= by_id.keys():
        raise ValueError('Prediction manifest contains empty, duplicate or unknown frame IDs')
    for frame_id in selected:
        if not isinstance(frame_id, str) or not frame_id or frame_id in ('.', '..') or '/' in frame_id or '\\' in frame_id:
            raise ValueError(f'Invalid frame ID: {frame_id!r}')
        if not (predictions / f'{frame_id}.npz').is_file():
            raise ValueError(f'Missing predictions for {frame_id}')
    return dataset, [by_id[frame_id] for frame_id in selected], run
