import cv2
import numpy as np


def save_overlay(image_path, output_path, truth, predicted, scores, matches, threshold):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot decode image: {image_path}')
    matched_pred = {p for p, _, _ in matches}
    matched_gt = {g for _, g, _ in matches}
    layers = [(truth, [(0, 180, 0) if i in matched_gt else (0, 200, 255) for i in range(len(truth['boxes']))], 'GT'),
              (predicted, [(255, 130, 0) if i in matched_pred else (0, 0, 255) for i in range(len(scores))], 'P')]
    for data, colors, prefix in layers:
        for i, (mask, box, color) in enumerate(zip(data['masks'], data['boxes'], colors)):
            if prefix == 'P' and scores[i] < threshold:
                continue
            image[mask] = (0.8 * image[mask] + 0.2 * np.array(color)).astype(np.uint8)
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 1)
            label = f'{prefix}{i}' if prefix == 'GT' else f'{prefix}{i} {scores[i]:.2f}'
            cv2.putText(image, label, (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(image, 'GT green | missed yellow | matched prediction blue | unmatched red',
                (8, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f'Cannot write overlay: {output_path}')
