from typing import Self

import cv2
import numpy as np
from PIL import Image

class MaskPostprocess:
    def __init__(self, *steps):
        self.steps = steps
        
    def __call__(self, mask):
        for step in self.steps:
            mask = step(mask)
        return mask


class MedianFilter:
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size
    
    def __call__(self, mask: np.ndarray):
        u8mask = mask.astype(np.uint8)
        u8mask = cv2.medianBlur(u8mask, self.kernel_size)
        return u8mask.astype(bool)


class Opening:
    def __init__(self, kernel_size: tuple[int, int]):
        self.kernel = np.ones(kernel_size)
    
    def __call__(self, mask: np.ndarray):
        u8mask = mask.astype(np.uint8)
        u8mask = cv2.morphologyEx(u8mask, cv2.MORPH_OPEN, self.kernel)
        return u8mask.astype(bool)


class Closing:
    def __init__(self, kernel_size: tuple[int, int]):
        self.kernel = np.ones(kernel_size)
    
    def __call__(self, mask: np.ndarray):
        u8mask = mask.astype(np.uint8)
        u8mask = cv2.morphologyEx(u8mask, cv2.MORPH_CLOSE, self.kernel)
        return u8mask.astype(bool)

class Dilate:
    def __init__(self, kernel_size: tuple[int, int]):
        self.kernel = np.ones(kernel_size)
        
    def __call__(self, mask: np.ndarray):
        u8mask = mask.astype(np.uint8)
        u8mask = cv2.dilate(u8mask, self.kernel)
        return u8mask.astype(bool)

    
class RegionOfInterestMasking:
    def __init__(self, roi_mask: np.ndarray):
        self.roi_mask = roi_mask
    
    def __call__(self, mask: np.ndarray):
        return mask * self.roi_mask
    
    @staticmethod
    def from_file(path: str) -> Self:
        roi_mask = np.array(Image.open(path))
        roi_mask = roi_mask > 100 # arbitrary threshold to get a binary mask
        return RegionOfInterestMasking(roi_mask.astype(np.uint8))


class RemoveSmallBlobs:
    def __init__(self, min_size: int):
        self.min_size = min_size
    
    def __call__(self, mask: np.ndarray):
        u8mask = mask.astype(np.uint8)
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(u8mask)
        
        sizes = stats[:, cv2.CC_STAT_AREA]
        resulting_mask = np.zeros_like(im_with_separated_blobs)
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] >= self.min_size:
                resulting_mask[im_with_separated_blobs == index_blob] = 255

        return resulting_mask.astype(bool)
