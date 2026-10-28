import argparse
import json
from pathlib import Path

from PIL import Image

from ml_ws.scripts.evaluation.data import load_instances, load_prediction_run
from ml_ws.segmentation.metrics import aggregate, box_iou, detection_stats, mask_iou
from ml_ws.scripts.evaluation.visualization import save_overlay


def main():
    parser = argparse.ArgumentParser(description='Evaluate class-agnostic visible-instance boxes and masks against generated ground truth.')
    parser.add_argument('--predictions', type=Path, required=True, help='Prediction directory under a model output run')
    parser.add_argument('--dataset', type=Path, help='Override dataset directory; otherwise read from the prediction run manifest')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1 or not 0 < args.iou_threshold <= 1:
        parser.error('Score threshold must be in [0, 1] and IoU threshold in (0, 1]')
    predictions = args.predictions.resolve()
    try:
        dataset, frames, prediction_manifest = load_prediction_run(predictions, args.dataset)
    except ValueError as error:
        parser.error(str(error))
    output = predictions.parent / 'evaluation'
    (output / 'overlays').mkdir(parents=True, exist_ok=True)
    samples = {'box': [], 'mask': []}
    per_image = []
    for frame in frames:
        image_path = dataset / frame['image']
        with Image.open(image_path) as image:
            shape = (image.height, image.width)
        truth = load_instances(dataset / frame['ground_truth'], shape)
        predicted = load_instances(predictions / f'{frame["id"]}.npz', shape, predictions=True)
        scores = predicted['scores']
        matrices = {'box': box_iou(predicted['boxes'], truth['boxes']),
                    'mask': mask_iou(predicted['masks'], truth['masks'])}
        entry = {'id': frame['id'], 'ground_truth_instances': len(truth['boxes']), 'predictions': len(scores)}
        for kind, matrix in matrices.items():
            samples[kind].append({'iou': matrix, 'scores': scores})
            entry[kind] = detection_stats(matrix, scores, args.score_threshold, args.iou_threshold)
        per_image.append(entry)
        save_overlay(image_path, output / 'overlays' / f'{frame["id"]}.png', truth, predicted,
                     scores, entry['mask']['matches'], args.score_threshold)
    report = {'evaluation': 'class_agnostic_visible_instances', 'run_status': prediction_manifest.get('status'),
              'split': prediction_manifest.get('split', 'all'), 'images': len(frames), 'score_threshold': args.score_threshold, 'iou_threshold': args.iou_threshold,
              'ap_protocol': 'Confidence-ranked matching, 101-point interpolation, IoU 0.50:0.05:0.95, max 100 predictions/image; no class or area splits. AP uses all saved scores.',
              'box_format': 'xyxy pixels, exclusive upper bounds',
              **{kind: aggregate(data, args.score_threshold, args.iou_threshold) for kind, data in samples.items()}}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'per_image.json').write_text(json.dumps(per_image, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
