
import numba
import numpy as np

from src.models.base import BackgroundModel
from src.models.still_models import bgr_to_gray, compute_bgr_mean
from src.video_source import VideoSource

from src.models.still_models import compute_gray_mean, compute_gray_variance_and_std, compute_gray_median

@numba.njit(parallel=True)
def adapt_step(mean, variance, background, gray_frame, frame_float, 
               alpha, std_bias, mean_rho, variance_rho):
    h, w = gray_frame.shape
    mask = np.empty((h, w), dtype=numba.boolean)
    one_minus_mrho = 1.0 - mean_rho
    one_minus_vrho = 1.0 - variance_rho
    
    for i in numba.prange(h):
        for j in range(w):
            g = gray_frame[i, j]
            m = mean[i, j]
            v = variance[i, j]
            s = np.sqrt(v)
            diff = abs(g - m)
            
            if diff >= alpha * (s + std_bias):
                mask[i, j] = True
            else:
                mask[i, j] = False
                # inplace update -> faster
                mean[i, j] = mean_rho * g + one_minus_mrho * m
                new_var = variance_rho * (diff * diff) + one_minus_vrho * v
                variance[i, j] = new_var
                for c in range(3):
                    background[i, j, c] = (mean_rho * frame_float[i, j, c] 
                                           + one_minus_mrho * background[i, j, c])
    
    return mask


class AdaptiveGrayGaussianModel(BackgroundModel):
    def __init__(self, alpha: float, mean_rho: float, variance_rho: float, std_bias: float=2.0):
        self.alpha = alpha
        self.mean_rho = mean_rho
        self.variance_rho = variance_rho
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

    """
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame)
        mask = np.abs(gray_frame - self.mean) >= self.alpha * (self.std + self.std_bias) # is the slides example supposed to be with u8 images??? then 2/255

        bg_mask = ~mask

        # adapt
        new_background = self.mean_rho * frame.astype(np.float32) + (1 - self.mean_rho) * self.background
        new_mean_for_background = self.mean_rho * gray_frame + (1 - self.mean_rho) * self.mean
        new_variance_for_background = self.variance_rho * np.square(gray_frame - self.mean) + (1 - self.variance_rho) * self.variance

        bg_mask_3d = bg_mask[:, :, np.newaxis]
        
        self.background = np.where(bg_mask_3d, new_background, self.background)
        self.mean = np.where(bg_mask, new_mean_for_background, self.mean)
        self.variance = np.where(bg_mask, new_variance_for_background, self.variance)

        self.std = np.sqrt(self.variance)

        return mask
    
    """
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = bgr_to_gray(frame).astype(np.float32)
        frame_float = frame.astype(np.float32)
        
        mask = adapt_step(
            self.mean, self.variance, self.background,
            gray_frame, frame_float,
            self.alpha, self.std_bias, self.mean_rho, self.variance_rho
        )
        
        return mask
    
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)
    
    def predict_from_source_with_frame(self, source):
        for frame in iter(source):
            yield self._predict_single(frame), frame
    
    def background_image(self):
        return self.background


# TODO: this is broken (no background)
class AdaptiveGrayMedianStdModel(BackgroundModel):
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
        new_background = self.mean_rho * frame.astype(np.float32) + (1 - self.mean_rho) * self.background
        new_mean_for_background = self.mean_rho * gray_frame + (1 - self.mean_rho) * self.median
        new_variance_for_background = self.variance_rho * np.square(gray_frame - self.median) + (1 - self.variance_rho) * self.variance
        
        bg_mask_3d = bg_mask[:, :, np.newaxis]
        mask_3d = mask[:, :, np.newaxis]
        
        self.background = bg_mask_3d * new_background + mask_3d * self.background
        self.median = bg_mask * new_mean_for_background + mask * self.median
        self.variance = bg_mask * new_variance_for_background + mask * self.variance
        
        self.std = np.sqrt(self.variance)
                
        return mask
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)
    
    def predict_from_source_with_frame(self, source):
        for frame in iter(source):
            yield self._predict_single(frame), frame
    
    def background_image(self):
        return self.background
