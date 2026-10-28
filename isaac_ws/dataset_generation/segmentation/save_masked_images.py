import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def save_masked_image(rgb, masks, instance_ids, destination):
    result = np.zeros_like(rgb[:, :, :3], dtype=np.uint8)
    for mask, instance_id in zip(masks, instance_ids):
        color = np.random.default_rng(int(instance_id)).integers(64, 256, size=3)
        result[mask] = (0.5 * rgb[:, :, :3][mask] + 0.5 * color).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(destination)


def main():
    parser = argparse.ArgumentParser(description='Save color-coded RGB instance-mask previews with a black background.')
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for frame in manifest['frames']:
        with Image.open(dataset / frame['image']) as image:
            rgb = np.array(image.convert('RGB'))
        with np.load(dataset / frame['ground_truth'], allow_pickle=False) as truth:
            save_masked_image(rgb, truth['masks'], truth['instance_ids'],
                              dataset / 'masked_images' / f'{frame["id"]}.png')
    print(f'Masked images: {dataset / "masked_images"}')


if __name__ == '__main__':
    main()
