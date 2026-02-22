
import numpy as np

from src.models.still_models import bgr_to_gray
from src.video_source import VideoSource


class AdaptiveGrayGaussianModel:
    def __init__(self, alpha: float, rho: float):
        self.alpha = alpha
        self.rho = rho
        self.mean = None
        self.variance = None
        self.std = None
    
    def _compute_mean(self, source: VideoSource) -> np.ndarray:
        frame_sum = 0
        frame_count = 0
        for frame in iter(source):
            gray_frame = bgr_to_gray(frame)
            frame_sum = frame_sum + gray_frame
            frame_count += 1
        
        return frame_sum / frame_count
    
    def _compute_variance_std(self, mean, source: VideoSource) -> np.ndarray:
        frame_variance_sum = 0
        frame_count = 0
        for frame in iter(source):
            gray_frame = bgr_to_gray(frame)
            frame_variance_sum = frame_variance_sum + np.square(gray_frame - mean)
            frame_count += 1
        
        variance = frame_variance_sum / (frame_count - 1)
        return variance, np.sqrt(variance)
    
    
    def fit_from_source(self, source: VideoSource):
        self.mean = self._compute_mean(source)
        self.variance, self.std = self._compute_variance_std(self.mean, source)
        return self
    
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame)
        mask = np.abs(gray_frame - self.mean) >= self.alpha * (self.std + 2) # is the slides example supposed to be with u8 images??? then 2/255
        
        bg_mask = ~mask
        
        # adapt
        new_mean_for_background = self.rho * gray_frame + (1 - self.rho) * self.mean
        new_variance_for_background = self.rho * np.square(gray_frame - self.mean) + (1 - self.rho) * self.variance
        
        self.mean = bg_mask * new_mean_for_background + mask * self.mean
        self.variance = bg_mask * new_variance_for_background + mask * self.variance
        
        self.std = np.sqrt(self.variance)
        
        # print(mask.shape); print(mask.ndim); exit()
        
        return mask
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)