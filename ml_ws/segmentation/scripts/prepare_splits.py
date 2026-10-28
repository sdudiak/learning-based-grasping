import argparse
from pathlib import Path

from ml_ws.segmentation.dataset import prepare_splits, read_manifest


def main():
    parser = argparse.ArgumentParser(description='Create disjoint pile-level train/validation/test manifests.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        if not read_manifest(args.dataset)[0].get('preparation'):
            parser.error('Run prepare-segmentation-dataset first, then split the prepared dataset')
        counts = prepare_splits(args.dataset.resolve(), args.validation_fraction, args.test_fraction, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f'Pile counts: {counts}; split manifests: {args.dataset / "splits"}')


if __name__ == '__main__':
    main()
