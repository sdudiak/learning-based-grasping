import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from ml_ws.segmentation.dataset import InstanceDataset, read_manifest
from ml_ws.segmentation.engine import seed_everything
from ml_ws.segmentation.model import ROOT
from ml_ws.segmentation.transforms import make_transforms


def prepare_dataset(source, output, config, copies=2, seed=42):
    if copies < 0:
        raise ValueError('Augmented copies must be nonnegative')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Prepared output must be empty: {output}')
    manifest, fingerprint = read_manifest(source)
    if manifest.get('preparation'):
        raise ValueError('Input must be an original generated dataset')
    seed_everything(seed)
    for directory in ('images', 'ground_truth', 'tensors'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    metadata = {'status': 'preparing', 'preparation': {'source': str(source.resolve()),
                'source_sha256': fingerprint, 'augmented_copies': copies, 'seed': seed,
                'augmentation': config['augmentation']}, 'frames': []}
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(metadata, indent=2) + '\n')
    original = InstanceDataset(source, manifest['frames'], make_transforms(config['augmentation']))
    augmented = InstanceDataset(source, manifest['frames'], make_transforms(config['augmentation'], training=True))
    for index, frame in enumerate(manifest['frames']):
        for variant in range(copies + 1):
            image, target = (original if variant == 0 else augmented)[index]
            name = frame['id'] if variant == 0 else f'{frame["id"]}_aug_{variant:03d}'
            record = {**frame, 'id': name, 'source_frame': frame['id'], 'augmented': variant != 0,
                      'image': f'images/{name}.png', 'ground_truth': f'ground_truth/{name}.npz',
                      'prepared': f'tensors/{name}.pt'}
            to_pil_image(image).save(output / record['image'])
            masks = target['masks'].numpy().astype(bool)
            np.savez_compressed(output / record['ground_truth'], masks=masks, instance_ids=np.arange(1, len(masks) + 1))
            torch.save({'image': (image * 255).round().to(torch.uint8), 'target': target}, output / record['prepared'])
            metadata['frames'].append(record)
        print(f'Prepared {index + 1}/{len(original)} source images', flush=True)
    metadata['status'] = 'complete'
    manifest_path.write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Save fixed augmentations and model-ready instance annotations before splitting.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--augmented-copies', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'configs/maskrcnn.toml')
    args = parser.parse_args()
    if not args.name or args.name in ('.', '..') or '/' in args.name or '\\' in args.name:
        parser.error('--name must be a single folder name')
    output = ROOT / 'ml_ws/datasets/segmentation' / args.name
    prepare_dataset(args.dataset.resolve(), output, tomllib.loads(args.config.read_text()), args.augmented_copies, args.seed)
    print(f'Prepared dataset: {output}')


if __name__ == '__main__':
    main()
