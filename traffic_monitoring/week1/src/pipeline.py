
from src.video_source import VideoSource

# TODO: preprocessing of the images? Maybe yes, maybe no
class DetectionPipeline:
    def __init__(self, background_model, mask_posprocess, detector):
        self.background_model = background_model
        self.mask_posprocess = mask_posprocess
        self.detector = detector
    
    def fit_from_source(self, source: VideoSource):
        self.background_model.fit_from_source(source)
        
    def predict_from_source(self, source: VideoSource):
        for mask in self.background_model.predict_from_source(source):
            postprocessed_mask = self.mask_posprocess(mask)
            detections = self.detector.detect(postprocessed_mask)
            yield detections
