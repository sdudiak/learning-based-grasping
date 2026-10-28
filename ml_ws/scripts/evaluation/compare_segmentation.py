import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np

from ml_ws.scripts.evaluation.data import load_instances, load_prediction_run
from ml_ws.segmentation.metrics import detection_stats, mask_iou


def comparison_image(image, truth, predicted, score_threshold=0.3, iou_threshold=0.5):
    stats = detection_stats(mask_iou(predicted['masks'], truth['masks']), predicted['scores'],
                            score_threshold, iou_threshold)
    colors = [tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb((i * 0.61803398875) % 1, 0.8, 1))
              for i in range(len(truth['masks']))]
    matched = {p: g for p, g, _ in stats['matches']}
    panels = []
    for instances, title, is_prediction in ((truth, 'Isaac ground truth', False),
                                             (predicted, 'Model prediction', True)):
        panel = image.copy()
        visible = 0
        for i, mask in enumerate(instances['masks']):
            if is_prediction and instances['scores'][i] < score_threshold:
                continue
            visible += 1
            if is_prediction:
                color = colors[matched[i]] if i in matched else (0, 0, 255)
            else:
                color = colors[i]
            panel[mask] = (0.45 * panel[mask] + 0.55 * np.array(color)).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, color, 1)
        panel = cv2.copyMakeBorder(panel, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(panel, f'{title}: {visible} instances', (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def main():
    parser = argparse.ArgumentParser(description='Compare Isaac and saved model masks side by side. Matched instances share colors; unmatched predictions are red.')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Defaults to evaluation/comparisons in the prediction run')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1 or not 0 < args.iou_threshold <= 1:
        parser.error('Score threshold must be in [0, 1] and IoU threshold in (0, 1]')
    predictions = args.predictions.resolve()
    try:
        dataset, frames, _ = load_prediction_run(predictions)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or predictions.parent / 'evaluation/comparisons'
    output.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        frame_id = frame['id']
        image = cv2.imread(str(dataset / frame['image']))
        if image is None:
            raise ValueError(f'Cannot decode image: {dataset / frame["image"]}')
        truth = load_instances(dataset / frame['ground_truth'], image.shape[:2])
        predicted = load_instances(predictions / f'{frame_id}.npz', image.shape[:2], predictions=True)
        comparison = comparison_image(image, truth, predicted, args.score_threshold, args.iou_threshold)
        if not cv2.imwrite(str(output / f'{frame_id}.png'), comparison):
            raise OSError(f'Cannot write comparison for {frame_id}')
    print(f'Saved {len(frames)} comparisons: {output}')


if __name__ == '__main__':
    main()
