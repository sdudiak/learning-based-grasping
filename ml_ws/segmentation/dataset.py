import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes


SPLITS = ('train', 'validation', 'test')


def read_manifest(dataset):
    path = Path(dataset) / 'manifest.json'
    raw = path.read_bytes()
    manifest = json.loads(raw)
    frames = manifest['frames']
    if manifest.get('status') != 'complete':
        raise ValueError('Dataset generation must be complete before preparing or using training splits')
    if not frames or any(not isinstance(f['id'], str) or f['id'] in ('.', '..') or '/' in f['id'] or '\\' in f['id'] for f in frames):
        raise ValueError('Dataset frame IDs must be nonempty file names')
    if any(not f['id'] for f in frames) or len({f['id'] for f in frames}) != len(frames):
        raise ValueError('Dataset must contain frames with unique IDs')
    if any('pile' not in frame for frame in frames):
        raise ValueError('Every frame must identify its pile')
    return manifest, hashlib.sha256(raw).hexdigest()


def prepare_splits(dataset, validation_fraction=0.1, test_fraction=0.1, seed=42):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError('Validation and test fractions must be positive and sum to less than one')
    manifest, fingerprint = read_manifest(dataset)
    piles = sorted({frame['pile'] for frame in manifest['frames']})
    if len(piles) < 3:
        raise ValueError('At least three captured piles are required for disjoint train/validation/test splits')
    random.Random(seed).shuffle(piles)
    n_validation = max(1, round(len(piles) * validation_fraction))
    n_test = max(1, round(len(piles) * test_fraction))
    if n_validation + n_test >= len(piles):
        raise ValueError('Requested fractions leave no training piles; use smaller fractions')
    groups = {'test': piles[:n_test], 'validation': piles[n_test:n_test + n_validation],
              'train': piles[n_test + n_validation:]}
    directory = Path(dataset) / 'splits'
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Splits already exist; retain them to keep experiments comparable: {directory}')
    for frame in manifest['frames']:
        for key in ('image', 'ground_truth'):
            if not (Path(dataset) / frame[key]).is_file():
                raise ValueError(f'Missing {key} for frame {frame["id"]}')
    directory.mkdir(parents=True, exist_ok=True)
    for name, ids in groups.items():
        split = {'dataset_manifest_sha256': fingerprint, 'seed': seed, 'pile_ids': ids,
                 'frame_ids': [f['id'] for f in manifest['frames'] if f['pile'] in ids and (name == 'train' or not f.get('augmented', False))]}
        (directory / f'{name}.json').write_text(json.dumps(split, indent=2) + '\n')
    return {name: len(group) for name, group in groups.items()}


def load_splits(dataset):
    manifest, fingerprint = read_manifest(dataset)
    frames = {frame['id']: frame for frame in manifest['frames']}
    seen_frames, seen_piles = set(), set()
    result = {}
    for name in SPLITS:
        split = json.loads((Path(dataset) / 'splits' / f'{name}.json').read_text())
        ids = split['frame_ids']
        if split['dataset_manifest_sha256'] != fingerprint:
            raise ValueError('Dataset manifest changed after splitting; use an unchanged dataset')
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= frames.keys():
            raise ValueError(f'Invalid frame IDs in {name} split')
        if name != 'train' and any(frames[frame_id].get('augmented', False) for frame_id in ids):
            raise ValueError('Validation and test splits must contain only original images')
        piles = {frames[frame_id]['pile'] for frame_id in ids}
        if seen_frames.intersection(ids) or seen_piles.intersection(piles):
            raise ValueError('Split leakage: frames or piles appear in more than one split')
        if piles != set(split['pile_ids']):
            raise ValueError(f'Pile metadata does not match frames in {name} split')
        seen_frames.update(ids)
        seen_piles.update(piles)
        result[name] = [frames[frame_id] for frame_id in ids]
    training_piles = {frame['pile'] for frame in result['train']}
    expected = {key for key, frame in frames.items()
                if not frame.get('augmented', False) or frame['pile'] in training_piles}
    if seen_frames != expected:
        raise ValueError('Splits must cover every captured frame exactly once')
    return result


def read_masks(root, frame, size):
    with np.load(root / frame['ground_truth'], allow_pickle=False) as data:
        masks = data['masks']
    if masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != (size[1], size[0]):
        raise ValueError(f'Invalid masks for frame {frame["id"]}')
    if len(masks) and not masks.any(axis=(1, 2)).all():
        raise ValueError(f'Empty instance mask in frame {frame["id"]}')
    return masks

class InstanceDataset(Dataset):
    def __init__(self, dataset, frames, transforms):
        self.root = Path(dataset)
        self.frames = frames
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        with Image.open(self.root / frame['image']) as image:
            image = image.convert('RGB')
        masks = read_masks(self.root, frame, image.size)
        image, masks = self.transforms(image, tv_tensors.Mask(torch.from_numpy(masks.astype(np.uint8))))
        boxes = masks_to_boxes(masks) if len(masks) else torch.zeros((0, 4), dtype=torch.float32)
        boxes[:, 2:] += 1  # masks_to_boxes returns inclusive maxima; training uses pixel-edge coordinates.
        return image, {'boxes': boxes, 'labels': torch.ones(len(masks), dtype=torch.int64),
                       'masks': masks, 'image_id': torch.tensor(index),
                       'area': masks.sum(dim=(1, 2)).float(), 'iscrowd': torch.zeros(len(masks), dtype=torch.int64)}


class PreparedDataset(Dataset):
    def __init__(self, dataset, frames):
        self.root = Path(dataset)
        self.frames = frames
        for frame in frames:
            if 'prepared' not in frame or not (self.root / frame['prepared']).is_file():
                raise ValueError('Run prepare-segmentation-dataset and split the prepared dataset before training')

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        data = torch.load(self.root / self.frames[index]['prepared'], map_location='cpu', weights_only=True)
        return data['image'].float().div_(255), data['target']


def collate(batch):
    return tuple(zip(*batch))
