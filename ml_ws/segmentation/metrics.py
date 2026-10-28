import numpy as np


def box_iou(predicted, truth):
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 4)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    low = np.maximum(predicted[:, None, :2], truth[None, :, :2])
    high = np.minimum(predicted[:, None, 2:], truth[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    pred_area = np.maximum(predicted[:, 2:] - predicted[:, :2], 0).prod(axis=1)
    gt_area = np.maximum(truth[:, 2:] - truth[:, :2], 0).prod(axis=1)
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def mask_iou(predicted, truth):
    result = np.zeros((len(predicted), len(truth)), dtype=np.float64)
    gt_area = truth.sum(axis=(1, 2), dtype=np.int64)
    for i, mask in enumerate(predicted):
        intersection = np.logical_and(truth, mask).sum(axis=(1, 2), dtype=np.int64)
        union = gt_area + mask.sum(dtype=np.int64) - intersection
        result[i] = np.divide(intersection, union, out=np.zeros(len(truth)), where=union > 0)
    return result


def match(iou, scores, threshold):
    """Match in confidence order, using each ground-truth instance at most once."""
    used = np.zeros(iou.shape[1], dtype=bool)
    matches = []
    for pred in np.argsort(-scores, kind='stable'):
        available = np.where(used, -1.0, iou[pred])
        if not len(available):
            continue
        gt = int(np.argmax(available))
        if available[gt] >= threshold:
            used[gt] = True
            matches.append((int(pred), gt, float(available[gt])))
    return matches


def detection_stats(iou, scores, score_threshold, iou_threshold):
    selected = np.flatnonzero(scores >= score_threshold)
    matches = match(iou[selected], scores[selected], iou_threshold)
    matches = [(int(selected[p]), g, value) for p, g, value in matches]
    tp = len(matches)
    fp = len(selected) - tp
    fn = iou.shape[1] - tp
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'mean_matched_iou': float(np.mean([v for _, _, v in matches])) if matches else None,
            'matches': matches}


def average_precision(samples, threshold, max_detections=100):
    total_gt = sum(sample['iou'].shape[1] for sample in samples)
    if total_gt == 0:
        return None
    outcomes = []
    for sample in samples:
        order = np.argsort(-sample['scores'], kind='stable')[:max_detections]
        scores = sample['scores'][order]
        matched = {p for p, _, _ in match(sample['iou'][order], scores, threshold)}
        outcomes.extend((float(score), int(i in matched)) for i, score in enumerate(scores))
    if not outcomes:
        return 0.0
    outcomes.sort(key=lambda pair: -pair[0])
    tp = np.cumsum([item[1] for item in outcomes])
    fp = np.cumsum([1 - item[1] for item in outcomes])
    recall = tp / total_gt
    precision = tp / (tp + fp)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    interpolated = [float(precision[recall >= level].max()) if np.any(recall >= level) else 0.0
                    for level in np.linspace(0, 1, 101)]
    return float(np.mean(interpolated))


def aggregate(samples, score_threshold, iou_threshold):
    stats = [detection_stats(s['iou'], s['scores'], score_threshold, iou_threshold) for s in samples]
    tp, fp, fn = (sum(s[key] for s in stats) for key in ('tp', 'fp', 'fn'))
    matched_ious = [value for s in stats for _, _, value in s['matches']]
    thresholds = np.linspace(0.5, 0.95, 10)
    ap = {f'{threshold:.2f}': average_precision(samples, threshold) for threshold in thresholds}
    defined = [value for value in ap.values() if value is not None]
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'mean_matched_iou': float(np.mean(matched_ious)) if matched_ious else None,
            'ap50': ap['0.50'], 'ap75': ap['0.75'],
            'ap50_95': float(np.mean(defined)) if defined else None, 'ap_by_iou': ap}
