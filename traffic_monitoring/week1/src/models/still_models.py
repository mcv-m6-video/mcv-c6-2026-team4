
import numpy as np

from src.video_source import VideoSource

def rgb_to_gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# TODO: preprocess the image with {avg, min, max} pooling? Maybe this removes the trees from the mask -> they move a bit (???)
# TODO: maybe downsizing the image (lowpass + downsize) does the same??????

class GrayGaussianModel:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mean = None
        self.variance = None
    
    def fit_from_source(self, source: VideoSource):
        frame_sum = 0
        frame_count = 0
        for frame in iter(source):
            gray_frame = rgb_to_gray(frame)
            frame_sum = frame_sum + gray_frame
            frame_count += 1
        
        self.mean = frame_sum / frame_count
        
        frame_variance_sum = 0
        frame_count = 0
        for frame in iter(source):
            gray_frame = rgb_to_gray(frame)
            frame_variance_sum = frame_variance_sum + np.square(gray_frame - self.mean)
            frame_count += 1
        
        self.variance = frame_variance_sum / (frame_count - 1)
        self.std = np.sqrt(self.variance)
        
        return self
    
    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = rgb_to_gray(frame)
        return np.abs(gray_frame - self.mean) >= self.alpha * (self.std + 2/255) # is the slides example supposed to be with u8 images??? then 2/255
    
    
    def predict_from_source(self, source: VideoSource):
        for frame in iter(source):
            yield self._predict_single(frame)
        

# TODO: color??????
