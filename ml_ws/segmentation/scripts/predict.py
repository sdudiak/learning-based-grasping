import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_ws.segmentation.dataset import InstanceDataset, PreparedDataset, collate, load_splits, read_manifest
from ml_ws.segmentation.model import build_model, output_directory, resolve_device
from ml_ws.segmentation.transforms import make_transforms


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Predict instance masks with a trained single-class Mask R-CNN checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--name')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dataset = (args.dataset or Path(checkpoint['dataset'])).resolve()
    manifest, fingerprint = read_manifest(dataset)
    if args.split == 'all':
        frames = manifest['frames']
    else:
        frames = load_splits(dataset)[args.split]
    output = output_directory(args.name or datetime.now(timezone.utc).strftime('maskrcnn_predictions_%Y%m%dT%H%M%S%fZ'))
    device = resolve_device(args.device)
    config = checkpoint['config']
    data = PreparedDataset(dataset, frames) if all('prepared' in frame for frame in frames) else InstanceDataset(dataset, frames, make_transforms(config['augmentation']))
    model = build_model(config['model'], initialize=False).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    (output / 'predictions').mkdir(parents=True)
    metadata = {'dataset': str(dataset), 'model': 'maskrcnn_resnet50_fpn_v2',
                'checkpoint': str(args.checkpoint.resolve()), 'split': args.split,
                'frame_ids': [frame['id'] for frame in frames], 'status': 'running',
                'device': str(device), 'classes': ['background', 'object'], 'dataset_manifest_sha256': fingerprint}
    def save():
        (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save()
    try:
        for index, (images, _) in enumerate(DataLoader(data, batch_size=1, collate_fn=collate)):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = model([images[0].to(device)])[0]
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            milliseconds = (time.perf_counter() - start) * 1000
            arrays = {'masks': result['masks'][:, 0].float().cpu().numpy() >= 0.5,
                      'boxes': result['boxes'].float().cpu().numpy(), 'scores': result['scores'].float().cpu().numpy(),
                      'labels': result['labels'].cpu().numpy()}
            frame = frames[index]
            np.savez_compressed(output / 'predictions' / f'{frame["id"]}.npz', **arrays)
            (output / 'predictions' / f'{frame["id"]}.json').write_text(json.dumps({
                'image': str(dataset / frame['image']), 'checkpoint': str(args.checkpoint.resolve()),
                'device': str(device), 'instances': len(arrays['scores']), 'latency_ms': milliseconds,
            }, indent=2) + '\n')
            print(f'{frame["id"]}: {len(arrays["scores"])} predictions, {milliseconds:.1f} ms', flush=True)
        metadata['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print(f'Predictions: {output / "predictions"}')


if __name__ == '__main__':
    main()
