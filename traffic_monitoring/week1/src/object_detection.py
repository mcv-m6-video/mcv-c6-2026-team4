
from collections import deque
from typing import NamedTuple, Optional

import cv2
import numpy as np


class BoundingBox(NamedTuple):
    top: float
    bottom: float
    left: float
    right: float
    confidence: Optional[float]
    

class CarDetector:
    def __init__(self, area: tuple[int, int], aspect_ratio: tuple[float, float], fill_ratio: tuple[float, float]):
        self.min_area = area[0] # TODO: max area too? max height or widdth? or something else?
        self.max_area = area[1]
        assert self.min_area <= self.max_area
        
        self.min_aspect_ratio = aspect_ratio[0]
        self.max_aspect_ratio = aspect_ratio[1]
        assert self.min_aspect_ratio <= self.max_aspect_ratio
        
        self.min_fill_ratio = fill_ratio[0]
        self.max_fill_ratio = fill_ratio[1]
        assert 0.0 <= self.min_fill_ratio <= self.max_fill_ratio <= 1.0

    
    def detect(self, mask: np.ndarray) -> list[BoundingBox]:
        u8mask = mask.astype(np.uint8)
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(u8mask)
        
        # TODO: the stats are: top left area height width max, maybe could use them for more refined detections? for instance, area / (width * height) > threshold
        
        boxes = []
        for index_blob in range(1, nb_blobs):
            area = stats[index_blob, cv2.CC_STAT_AREA]
            height = stats[index_blob, cv2.CC_STAT_HEIGHT]
            width = stats[index_blob, cv2.CC_STAT_WIDTH]
            
            fill = area / (height * width)
            aspect_ratio = width / height
            
            area_ok = self.min_area <= area <= self.max_area
            fill_ok = self.min_fill_ratio <= fill <= self.max_fill_ratio
            aspect_ratio_ok = self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            
            if area_ok and fill_ok and aspect_ratio_ok:
                top = stats[index_blob, cv2.CC_STAT_TOP]
                bottom = height + top
                left = stats[index_blob, cv2.CC_STAT_LEFT]
                right = width + left
                boxes.append(BoundingBox(
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                    confidence=fill
                ))
        
        return boxes


class TemporalCarDetector:
    def __init__(self, base_detector: CarDetector, n_frames: int = 5, threshold: float = 0.5):
        self.base_detector = base_detector
        self.n_frames = n_frames
        self.threshold = threshold
        self.mask_buffer: deque[np.ndarray] = deque(maxlen=n_frames)

    def detect(self, mask: np.ndarray) -> list[BoundingBox]:
        self.mask_buffer.append(mask.astype(np.float32))
        weights = np.arange(1, len(self.mask_buffer) + 1, dtype=np.float32)
        weights /= weights.sum()
        accumulated = sum(w * m for w, m in zip(weights, self.mask_buffer))
        smoothed_mask = accumulated >= self.threshold
        return self.base_detector.detect(smoothed_mask)