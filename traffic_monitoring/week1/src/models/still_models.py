
from typing import Tuple
import numpy as np
import cv2

from src.models.base import BackgroundModel
from src.video_source import VideoSource

def rgb_to_gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def bgr_to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# TODO: preprocess the image with {avg, min, max} pooling? Maybe this removes the trees from the mask -> they move a bit (???)
# TODO: maybe downsizing the image (lowpass + downsize) does the same??????



def compute_gray_mean(source: VideoSource) -> np.ndarray:
        frame_sum = 0
        frame_count = 0
        for frame in iter(source):
            gray_frame = bgr_to_gray(frame).astype(np.float32)
            frame_sum = frame_sum + gray_frame
            frame_count += 1
        
        return frame_sum / frame_count


def compute_bgr_mean(source: VideoSource) -> np.ndarray:
        frame_sum = 0
        frame_count = 0
        for frame in iter(source):
            frame_sum = frame_sum + frame.astype(np.float32)
            frame_count += 1
        
        return frame_sum / frame_count


def compute_gray_variance_and_std(mean: np.ndarray, source: VideoSource) -> Tuple[np.ndarray, np.ndarray]:
        frame_variance_sum = 0
        frame_count = 0
        for frame in iter(source):
            gray_frame = bgr_to_gray(frame).astype(np.float32)
            frame_variance_sum = frame_variance_sum + np.square(gray_frame - mean)
            frame_count += 1
        
        variance = frame_variance_sum / (frame_count - 1)
        std = np.sqrt(variance)
        
        return variance, std


def compute_gray_median(source: VideoSource) -> np.ndarray:
    gray_frames = np.stack([bgr_to_gray(frame) for frame in iter(source)], axis=-1)
    return np.median(gray_frames, axis=2).astype(np.float32)


class GrayGaussianModel(BackgroundModel):
    def __init__(self, alpha: float, std_bias: float=2.0):
        self.alpha = alpha
        self.mean = None
        self.variance = None
        self.background = None
        self.std = None
        self.std_bias = std_bias
    
    def fit_from_source(self, source: VideoSource):
        self.background = compute_bgr_mean(source)
        self.mean = bgr_to_gray(self.background)
        self.variance, self.std = compute_gray_variance_and_std(self.mean, source)
        return self
    
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame).astype(np.float32)
        return np.abs(gray_frame - self.mean) >= self.alpha * (self.std + self.std_bias) # is the slides example supposed to be with u8 images??? then 2/255
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)
    
    def predict_from_source_with_frame(self, source):
        for frame in iter(source):
            yield self._predict_single(frame), frame
    
    def background_image(self):
        return self.background


class GrayMedianStdModel(BackgroundModel):
    def __init__(self, alpha: float, std_bias: float=2.0):
        self.alpha = alpha
        self.median = None
        self.variance = None
        self.background = None
        self.std = None
        self.std_bias = std_bias
    
    def fit_from_source(self, source: VideoSource):
        self.background = compute_bgr_mean(source)
        self.median = bgr_to_gray(self.background)
        self.variance, self.std = compute_gray_variance_and_std(self.median, source)
        return self

    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame).astype(np.float32)
        return np.abs(gray_frame - self.median) >= self.alpha * (self.std + self.std_bias) # is the slides example supposed to be with u8 images??? then 2/255
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)
    
    def predict_from_source_with_frame(self, source):
        for frame in iter(source):
            yield self._predict_single(frame), frame
    
    def background_image(self):
        return self.background

# TODO: color??????
