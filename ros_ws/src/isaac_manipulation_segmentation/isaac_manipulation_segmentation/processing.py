from threading import Condition, Thread

import cv2
import numpy as np


class FrameProcessor:
    def __init__(self, decimation, process):
        if type(decimation) is not int or decimation < 1:
            raise ValueError('decimation must be an integer >= 1')
        self.decimation = decimation
        self.process = process
        self.received = 0
        self.pending = None
        self.stopped = False
        self.condition = Condition()
        self.thread = Thread(target=self._run, name='segmentation-inference')
        self.thread.start()

    def submit(self, frame):
        with self.condition:
            selected = self.received % self.decimation == 0
            self.received += 1
            if selected and not self.stopped:
                self.pending = frame
                self.condition.notify()

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify()
        self.thread.join()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                if self.stopped:
                    return
                frame, self.pending = self.pending, None
            self.process(frame)


def instance_mask(masks, scores):
    if len(masks) > np.iinfo(np.uint16).max:
        raise ValueError('Too many instances for a 16-bit instance mask')
    image = np.zeros(masks.shape[1:], dtype=np.uint16)
    for index in np.argsort(-scores, kind='stable'):
        image[masks[index] & (image == 0)] = index + 1
    return image


class OverlayTracker:
    def __init__(self, iou_threshold, max_missed_frames):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes, labels):
        boxes = np.asarray(boxes).reshape(-1, 4)
        labels = np.asarray(labels)
        ids = np.zeros(len(boxes), dtype=np.int64)
        for track in self.tracks.values():
            track['missed'] += 1
        if self.tracks and len(boxes):
            track_ids = list(self.tracks)
            previous = np.array([self.tracks[key]['box'] for key in track_ids])
            low = np.maximum(boxes[:, None, :2], previous[None, :, :2])
            high = np.minimum(boxes[:, None, 2:], previous[None, :, 2:])
            intersection = np.maximum(high - low, 0).prod(axis=2)
            areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
            previous_areas = np.maximum(previous[:, 2:] - previous[:, :2], 0).prod(axis=1)
            union = areas[:, None] + previous_areas[None, :] - intersection
            iou = intersection / np.maximum(union, 1e-8)
            previous_labels = np.array([self.tracks[key]['label'] for key in track_ids])
            matches = np.argwhere((iou >= self.iou_threshold) & (labels[:, None] == previous_labels))
            used = set()
            for detection, track in sorted(matches, key=lambda pair: -iou[tuple(pair)]):
                if ids[detection] == 0 and track not in used:
                    ids[detection] = track_ids[track]
                    used.add(track)
        for index, box in enumerate(boxes):
            if ids[index] == 0:
                ids[index] = self.next_id
                self.next_id += 1
            self.tracks[int(ids[index])] = {'box': box.copy(), 'label': labels[index], 'missed': 0}
        self.tracks = {key: track for key, track in self.tracks.items()
                       if track['missed'] <= self.max_missed_frames}
        return ids


def render_overlay(rgb, instances, boxes, scores, color_ids=None):
    image = rgb.copy()
    for index, (box, score) in enumerate(zip(boxes, scores), start=1):
        color_id = int(color_ids[index - 1]) if color_ids is not None else index
        color = ((color_id * 67) % 192 + 63, (color_id * 113) % 192 + 63, (color_id * 157) % 192 + 63)
        mask = instances == index
        image[mask] = (image[mask].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        x1, y1 = np.floor(box[:2]).astype(int)
        x2, y2 = np.ceil(box[2:]).astype(int) - 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{index}: {score:.2f}', (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return image
