
from typing import NamedTuple

import cv2
import numpy as np


class BoundingBox(NamedTuple):
    top: float
    bottom: float
    left: float
    right: float
    

class CarDetector:
    def __init__(self, min_area: int):
        self.min_area = min_area # TODO: max area too? max height or widdth? or something else?
    
    def detect(self, mask: np.ndarray) -> list[BoundingBox]:
        u8mask = mask.astype(np.uint8)
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(u8mask)
        
        # TODO: the stats are: top left area height width max, maybe could use them for more refined detections? for instance, area / (width * height) > threshold
        
        boxes = []
        for index_blob in range(1, nb_blobs):
            size = stats[index_blob, cv2.CC_STAT_AREA]
            if size >= self.min_area:
                top = stats[index_blob, cv2.CC_STAT_TOP]
                bottom = stats[index_blob, cv2.CC_STAT_HEIGHT] + top
                left = stats[index_blob, cv2.CC_STAT_LEFT]
                right = stats[index_blob, cv2.CC_STAT_WIDTH] + left
                boxes.append(BoundingBox(
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right
                ))
        
        return boxes
        