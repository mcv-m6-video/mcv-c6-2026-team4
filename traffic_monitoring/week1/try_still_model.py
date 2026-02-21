
import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from src.object_detection import BoundingBox, CarDetector
from src.models.still_models import GrayGaussianModel
from src.video_source import VideoPartSource
from src.mask_postprocessing import MaskPostprocess, MedianFilter, RemoveSmallBlobs
import cv2

def draw_bboxes(image, bboxes: list[BoundingBox]):
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    VIDEO_PATH = "/home/arnau-marcos-almansa/Downloads/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    OUTPUT_PATH = "predicted_masks.avi"
    
    train_source = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
    test_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)
    
    model = GrayGaussianModel(2.5)
    
    print("Fitting")
    model.fit_from_source(train_source)
    
    # TODO: for the moment these are arbitrary hyperparameters
    postprocess = MaskPostprocess(
        MedianFilter(3),
        RemoveSmallBlobs(300),
    )
    
    detector = CarDetector(1000)
    
    mask_writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"XVID"),
        test_source.fps,
        (test_source.width, test_source.height),
        isColor=False,
    )
    
    detection_writer = cv2.VideoWriter(
        "predicted_detections.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        test_source.fps,
        (test_source.width, test_source.height),
    )
    
    
    print("Predicting")
    for mask, og_frame in tqdm(zip(model.predict_from_source(test_source), iter(test_source)), total=test_source.n_frames): # bastante F hacer esto
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        mask = postprocess(mask)
        bboxes = detector.detect(mask)
                
        if mask.dtype != np.uint8:
            mask = (mask.astype(bool).astype(np.uint8)) * 255

        og_frame_hwc = np.ascontiguousarray(og_frame.transpose((1, 2, 0)))
        detections_frame = draw_bboxes(og_frame_hwc, bboxes)
        detections_frame = cv2.cvtColor(detections_frame, cv2.COLOR_RGB2BGR)

        mask_writer.write(mask)
        detection_writer.write(detections_frame)

    mask_writer.release()
    detection_writer.release()