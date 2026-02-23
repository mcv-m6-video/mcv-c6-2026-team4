
import cv2
import numpy as np
from tqdm import tqdm
from src.models.still_models import GrayGaussianModel
from src.bbox_merging import PolBoundingBoxMerger
from src.shadow_removal import HSVBackgroundComparison, NoShadowRemoval
from src.object_detection import TemporalCarDetector, CarDetector
from src.mask_postprocessing import MaskPostprocess, Opening, Closing, Dilate, RemoveSmallBlobs
from src.evaluation import evaluate_detections, load_annotations, show_metrics
from src.models.adaptive_models import AdaptiveGrayGaussianModel
from src.video_source import VideoPartSource
from src.pipeline import DetectionPipeline
from try_still_model import draw_bboxes
import cProfile
import pstats
from io import StringIO


if __name__ == '__main__':
    ANNOTATIONS_PATH = "../ai_challenge_s03_c010-full_annotation.xml"
    VIDEO_PATH = "../AICity_data/AICity_data/train/S03/c010/vdo.avi"
    ROI_PATH = "../AICity_data/AICity_data/train/S03/c010/roi.jpg"
    OUTPUT_PATH = "predicted_masks.avi"
    
    train_source = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
    test_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)
    annotations = load_annotations(ANNOTATIONS_PATH)
    
    model = AdaptiveGrayGaussianModel(3.0, mean_rho=0.05, variance_rho=0.05)
    # model = GrayGaussianModel(3.0)
    
    # TODO: for the moment these are arbitrary hyperparameters
    postprocess = MaskPostprocess(
        Opening((5, 5)),
        Closing((20, 20)),
        Dilate((7, 7)),
        RemoveSmallBlobs(300),
    )
    
    detector = TemporalCarDetector(
        base_detector=CarDetector(
            area=[1000, 10000000000],
            aspect_ratio=[0.1, 10.0],
            fill_ratio=[0.2, 1.0]
        ),
        n_frames=3,
        threshold=0.5
    )
    
    pipeline = DetectionPipeline(
        background_model=model,
        shadow_remover=HSVBackgroundComparison(),
        mask_posprocess=postprocess,
        detector=detector,
        bbox_merger=PolBoundingBoxMerger(merge_distance=40)
    )

    print("Fitting")
    pipeline.fit_from_source(train_source)
    

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

    side_by_side_writer = cv2.VideoWriter(
        "side_by_side.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        test_source.fps,
        (test_source.width * 2, test_source.height),
    )
    
    
    predictions = {}

    print("Predicting")
    masks_and_frames = zip(
        pipeline.predict_from_source_with_extras(test_source),
        range(test_source.start_frame, test_source.end_frame) # TODO: +1???
    )
    
    for (detections, extras), frame_id in tqdm(masks_and_frames, total=test_source.n_frames): # bastante F hacer esto
        predictions[frame_id] = detections
        
        mask = extras["postprocessed_mask"]
        if mask.dtype != np.uint8:
            mask = (mask.astype(bool).astype(np.uint8)) * 255

        detections_frame = draw_bboxes(extras["frame"], detections, (0, 255, 0))
        if frame_id in annotations:
            detections_frame = draw_bboxes(detections_frame, annotations[frame_id], (255, 0, 0))
        
        mask_writer.write(mask)
        detection_writer.write(detections_frame)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        side_by_side_writer.write(np.concatenate([mask_bgr, detections_frame], axis=1))

    mask_writer.release()
    detection_writer.release()
    side_by_side_writer.release()

    gt_for_eval = {fid: boxes for fid, boxes in annotations.items() if fid in predictions}
    metrics = evaluate_detections(gt_for_eval, predictions)
    show_metrics(metrics)
