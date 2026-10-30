import numpy as np


class Selection:
    def __init__(self, threshold, association_threshold, ambiguity):
        self.threshold = threshold
        self.association_threshold = association_threshold
        self.ambiguity = ambiguity
        self.reference = None

    def nearest(self, candidates, reference, threshold):
        ranked = sorted((float(np.linalg.norm(center - reference)), label)
                        for label, center in candidates.items())
        if not ranked or ranked[0][0] > threshold:
            return None
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= self.ambiguity:
            return None
        return ranked[0][1]

    def select(self, candidates, reference):
        if not np.isfinite(reference).all():
            return None
        label = self.nearest(candidates, reference, self.threshold)
        if label is not None:
            self.reference = candidates[label].copy()
        return label

    def match(self, candidates):
        if self.reference is None:
            return None
        return self.nearest(candidates, self.reference, self.association_threshold)

    def clear(self):
        self.reference = None
