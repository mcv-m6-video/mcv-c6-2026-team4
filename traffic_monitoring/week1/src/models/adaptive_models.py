
import numpy as np

from src.models.still_models import bgr_to_gray
from src.video_source import VideoSource

from src.models.still_models import compute_gray_mean, compute_gray_variance_and_std, compute_gray_median


class AdaptiveGrayGaussianModel:
    def __init__(self, alpha: float, mean_rho: float, variance_rho: float, std_bias: float=2.0):
        self.alpha = alpha
        self.mean_rho = mean_rho
        self.variance_rho = variance_rho
        self.mean = None
        self.variance = None
        self.std = None
        self.std_bias = std_bias

    
    def fit_from_source(self, source: VideoSource):
        self.mean = compute_gray_mean(source)
        self.variance, self.std = compute_gray_variance_and_std(self.mean, source)
        return self

    
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame)
        mask = np.abs(gray_frame - self.mean) >= self.alpha * (self.std + self.std_bias) # is the slides example supposed to be with u8 images??? then 2/255
        
        bg_mask = ~mask
        
        # adapt
        new_mean_for_background = self.mean_rho * gray_frame + (1 - self.mean_rho) * self.mean
        new_variance_for_background = self.variance_rho * np.square(gray_frame - self.mean) + (1 - self.variance_rho) * self.variance
        
        self.mean = bg_mask * new_mean_for_background + mask * self.mean
        self.variance = bg_mask * new_variance_for_background + mask * self.variance
        
        self.std = np.sqrt(self.variance)
                
        return mask
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)


class AdaptiveGrayMedianStdModel:
    def __init__(self, alpha: float, mean_rho: float, variance_rho: float, std_bias: float=2.0):
        self.alpha = alpha
        self.mean_rho = mean_rho
        self.variance_rho = variance_rho
        self.median = None
        self.variance = None
        self.std = None
        self.std_bias = std_bias

    
    def fit_from_source(self, source: VideoSource):
        self.median = compute_gray_median(source)
        self.variance, self.std = compute_gray_variance_and_std(self.median, source)
        return self

    
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame)
        mask = np.abs(gray_frame - self.median) >= self.alpha * (self.std + self.std_bias) # is the slides example supposed to be with u8 images??? then 2/255
        
        bg_mask = ~mask
        
        # adapt
        new_mean_for_background = self.mean_rho * gray_frame + (1 - self.mean_rho) * self.median
        new_variance_for_background = self.variance_rho * np.square(gray_frame - self.median) + (1 - self.variance_rho) * self.variance
        
        self.median = bg_mask * new_mean_for_background + mask * self.median
        self.variance = bg_mask * new_variance_for_background + mask * self.variance
        
        self.std = np.sqrt(self.variance)
                
        return mask
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)
