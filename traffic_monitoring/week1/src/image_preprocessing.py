
import math

import cv2
import numpy as np


class ImagePreprocess:
    def __init__(self, *steps):
        self.steps = steps
        
    def __call__(self, image):
        for step in self.steps:
            image = step(image)
        return image

class GaussianBlur:
    def __init__(self, sigma: float):
        self.sigma = sigma
    
    def __call__(self, image: np.ndarray):
        ksize = (math.ceil(self.sigma * 8), math.ceil(self.sigma * 8))
        return cv2.GaussianBlur(image, ksize, self.sigma)
