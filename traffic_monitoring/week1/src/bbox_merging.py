
from typing import Protocol

from src.object_detection import BoundingBox


class BoundingBoxMerger(Protocol):
    def merge(self, bboxes: list[BoundingBox]):
        pass


class NoBoundingBoxMerge(BoundingBoxMerger):
    def merge(self, bboxes):
        return bboxes


class PolBoundingBoxMerger(BoundingBoxMerger):
    def __init__(self, merge_distance: float):
        super().__init__()
        self.merge_distance = merge_distance
    
    def merge(self, bboxes):
        """Merge bounding boxes whose edges are within merge_distance pixels of each other."""
        if len(bboxes) < 2:
            return bboxes

        # Union-Find
        parent = list(range(len(bboxes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        def bbox_gap(a: BoundingBox, b: BoundingBox) -> float:
            dx = max(0.0, max(a.left - b.right, b.left - a.right))
            dy = max(0.0, max(a.top - b.bottom, b.top - a.bottom))
            return (dx**2 + dy**2) ** 0.5

        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if bbox_gap(bboxes[i], bboxes[j]) <= self.merge_distance:
                    union(i, j)

        groups: dict[int, list[BoundingBox]] = {}
        for i, box in enumerate(bboxes):
            root = find(i)
            groups.setdefault(root, []).append(box)

        merged = []
        for group in groups.values():
            top = min(b.top for b in group)
            bottom = max(b.bottom for b in group)
            left = min(b.left for b in group)
            right = max(b.right for b in group)
            confidence = max((b.confidence for b in group if b.confidence is not None), default=None)
            merged.append(BoundingBox(top=top, bottom=bottom, left=left, right=right, confidence=confidence))

        return merged