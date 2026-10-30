from collections import OrderedDict

import numpy as np


def project_depth(depth, k, minimum=0.01, maximum=10.0):
    depth = np.asarray(depth, dtype=np.float32)
    k = np.asarray(k).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    depth = np.where(np.isfinite(depth), depth, np.nan)
    v, u = np.indices(depth.shape)
    points = np.stack(((u - k[0, 2]) * depth / k[0, 0],
                       (v - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
    valid = np.isfinite(points).all(axis=-1) & (depth > 0) & (depth >= minimum) & (depth <= maximum)
    return points, valid


def transform_points(points, translation, quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError('Invalid quaternion')
    x, y, z, w = q / norm
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return points @ rotation.T + np.asarray(translation)


def candidates(points, labels, minimum):
    return {int(label): points[labels == label].mean(axis=0)
            for label in np.unique(labels) if label and np.count_nonzero(labels == label) >= minimum}


def partition(points, labels, target=None):
    selected = labels == target if target is not None else np.zeros(len(points), dtype=bool)
    return points[selected], points[~selected]


def bounds(points):
    low, high = points.min(axis=0), points.max(axis=0)
    return points.mean(axis=0), (low + high) / 2, high - low


class ObservationCache:
    def __init__(self, capacity, age, tolerance=0):
        self.capacity, self.age, self.tolerance = capacity, age, tolerance
        self.items = OrderedDict()

    def add(self, stamp, value):
        if self.items and stamp < max(self.items) - self.age:
            self.items.clear()
        self.items[stamp] = value
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        for key in list(self.items):
            if key < stamp - self.age:
                del self.items[key]

    def nearest(self, stamp):
        if not self.items:
            return None
        key = min(self.items, key=lambda key: abs(key - stamp))
        return self.items[key] if abs(key - stamp) <= self.tolerance else None
