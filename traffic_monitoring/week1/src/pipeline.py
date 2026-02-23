
from src.bbox_merging import BoundingBoxMerger
from src.models.base import BackgroundModel
from src.shadow_removal import ShadowRemover
from src.video_source import VideoSource

# TODO: preprocessing of the images? Maybe yes, maybe no
class DetectionPipeline:
    def __init__(self, *, background_model: BackgroundModel, shadow_remover: ShadowRemover, mask_posprocess, detector, bbox_merger: BoundingBoxMerger):
        self.background_model = background_model
        self.shadow_remover = shadow_remover
        self.mask_posprocess = mask_posprocess
        self.detector = detector
        self.bbox_merger = bbox_merger
    
    def fit_from_source(self, source: VideoSource):
        self.background_model.fit_from_source(source)
    
    def predict_from_source(self, source: VideoSource):
        for final_detections, _ in self.predict_from_source_with_extras(source):
            yield final_detections
    
    def predict_from_source_with_extras(self, source: VideoSource):
        for mask, frame in self.background_model.predict_from_source_with_frame(source):
            mask_without_shadows = self.shadow_remover.remove_shadows(mask, frame, self.background_model)
            postprocessed_mask = self.mask_posprocess(mask_without_shadows)
            detections = self.detector.detect(postprocessed_mask)
            merged_detections = self.bbox_merger.merge(detections)
            
            extras = {
                "frame": frame,
                "mask": mask,
                "mask_without_shadows": mask_without_shadows,
                "postprocessed_mask": postprocessed_mask,
                "detections": detections,
            }
            
            yield merged_detections, extras
