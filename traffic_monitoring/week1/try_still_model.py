
import matplotlib


matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from src.object_detection import BoundingBox, CarDetector
from src.models.still_models import GrayGaussianModel
from src.models.adaptive_models import AdaptiveGrayGaussianModel
from src.video_source import VideoPartSource
from src.mask_postprocessing import Closing, Dilate, MaskPostprocess, MedianFilter, Opening, RegionOfInterestMasking, RemoveSmallBlobs
from src.evaluation import evaluate_detections, show_metrics
import cv2

import xml.etree.ElementTree as ET


def draw_bboxes(image, bboxes: list[BoundingBox], color: tuple[int, int, int]):
    for bbox in bboxes:
        image = cv2.rectangle(image, (round(bbox.left), round(bbox.top)), (round(bbox.right), round(bbox.bottom)), color, 2)
    return image

def load_annotations(path: str):
    tree = ET.parse(path)
    boxes = tree.findall("track/box")
    boxes_per_frame = dict()
    for box in boxes:
        frame    = int(box.get("frame"))
        xtl      = float(box.get("xtl"))
        ytl      = float(box.get("ytl"))
        xbr      = float(box.get("xbr"))
        ybr      = float(box.get("ybr"))
        outside  = box.get("outside") == "1"
        occluded = box.get("occluded") == "1"
        keyframe = box.get("keyframe") == "1"

        parked = None
        for attr in box.findall("attribute"):
            if attr.get("name") == "parked":
                parked = attr.text == "true"
                
        # TODO: should we even discard parked cars?
        if not parked:
            bbox = BoundingBox(
                top=ytl,
                bottom=ybr,
                left=xtl,
                right=xbr,
                confidence=1.0
            )
            
            if frame not in boxes_per_frame:
                boxes_per_frame[frame] = [bbox]
            else:
                boxes_per_frame[frame].append(bbox)
    
    return boxes_per_frame


if __name__ == '__main__':
    ANNOTATIONS_PATH = "/home/arnau-marcos-almansa/Downloads/ai_challenge_s03_c010-full_annotation.xml"
    VIDEO_PATH = "/home/arnau-marcos-almansa/Downloads/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    ROI_PATH = "/home/arnau-marcos-almansa/Downloads/AICity_data/AICity_data/train/S03/c010/roi.jpg"
    OUTPUT_PATH = "predicted_masks.avi"
    
    train_source = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
    test_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)
    
    model = AdaptiveGrayGaussianModel(3.0, mean_rho=0.05, variance_rho=0.05)
    
    annotations = load_annotations(ANNOTATIONS_PATH)
    
    print("Fitting")
    model.fit_from_source(train_source)
    
    # TODO: for the moment these are arbitrary hyperparameters
    postprocess = MaskPostprocess(
        # MedianFilter(3),
        # RegionOfInterestMasking.from_file(ROI_PATH),
        Opening((5, 5)),
        Closing((5, 5)),
        Dilate((7, 7)),
        RemoveSmallBlobs(300),
    )
    
    detector = CarDetector(
        area=[1000, 100000],
        aspect_ratio=[0.1, 10.0],
        fill_ratio=[0.2, 1.0]
    )
    
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
    
    
    predictions = {}

    print("Predicting")
    masks_and_frames = zip(
        model.predict_from_source(test_source),
        enumerate(iter(test_source), start=test_source.start_frame),
    )
    
    for mask, (frame_id, og_frame) in tqdm(masks_and_frames, total=test_source.n_frames): # bastante F hacer esto
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        mask = postprocess(mask)
        bboxes = detector.detect(mask)
        predictions[frame_id] = bboxes
                
        if mask.dtype != np.uint8:
            mask = (mask.astype(bool).astype(np.uint8)) * 255

        detections_frame = draw_bboxes(og_frame, bboxes, (0, 255, 0))
        if frame_id in annotations:
            detections_frame = draw_bboxes(detections_frame, annotations[frame_id], (255, 0, 0))
        
        mask_writer.write(mask)
        detection_writer.write(detections_frame)

    mask_writer.release()
    detection_writer.release()

    gt_for_eval = {fid: boxes for fid, boxes in annotations.items() if fid in predictions}
    metrics = evaluate_detections(gt_for_eval, predictions)
    show_metrics(metrics)
