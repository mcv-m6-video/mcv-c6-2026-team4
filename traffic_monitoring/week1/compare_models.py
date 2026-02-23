"""
Comparison script for different background subtraction models.

Compares custom implementations (GrayGaussianModel, AdaptiveGrayGaussianModel)
with standard OpenCV methods (MOG, MOG2, LSBP).
"""

import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

from src.object_detection import BoundingBox, CarDetector, TemporalCarDetector, merge_bboxes
from src.models.still_models import GrayGaussianModel, GrayMedianStdModel
from src.models.adaptive_models import AdaptiveGrayGaussianModel, AdaptiveGrayMedianStdModel
from src.models.opencv_models import MOGBackgroundSubtractor, MOG2BackgroundSubtractor, LSBPBackgroundSubtractor
from src.video_source import VideoPartSource
from src.mask_postprocessing import Closing, Dilate, MaskPostprocess, Opening, RemoveSmallBlobs
from src.evaluation import evaluate_detections, show_metrics


def load_annotations(path: str) -> Dict[int, List[BoundingBox]]:
    """Load ground truth annotations from XML file."""
    tree = ET.parse(path)
    boxes = tree.findall("track/box")
    boxes_per_frame = dict()

    for box in boxes:
        frame = int(box.get("frame"))
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        parked = None
        for attr in box.findall("attribute"):
            if attr.get("name") == "parked":
                parked = attr.text == "true"

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


def draw_bboxes(image: np.ndarray, bboxes: List[BoundingBox], color: Tuple[int, int, int]) -> np.ndarray:
    """Draw bounding boxes on an image."""
    for bbox in bboxes:
        image = cv2.rectangle(
            image,
            (round(bbox.left), round(bbox.top)),
            (round(bbox.right), round(bbox.bottom)),
            color,
            2
        )
    return image


def evaluate_model(model, model_name: str, train_source: VideoPartSource,
                   test_source: VideoPartSource, annotations: Dict[int, List[BoundingBox]],
                   postprocess: MaskPostprocess, detector,
                   save_videos: bool = False) -> Dict:
    """
    Evaluate a single background subtraction model.

    Args:
        model: Background subtraction model
        model_name: Name of the model for output files
        train_source: Video source for training
        test_source: Video source for testing
        annotations: Ground truth annotations
        postprocess: Mask postprocessing pipeline
        detector: Car detector
        save_videos: Whether to save output videos

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Training phase
    print("Training...")
    model.fit_from_source(train_source)

    # Setup video writers if needed
    mask_writer = None
    detection_writer = None
    side_by_side_writer = None

    if save_videos:
        mask_writer = cv2.VideoWriter(
            f"{model_name}_masks.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            test_source.fps,
            (test_source.width, test_source.height),
            isColor=False,
        )

        detection_writer = cv2.VideoWriter(
            f"{model_name}_detections.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            test_source.fps,
            (test_source.width, test_source.height),
        )

        side_by_side_writer = cv2.VideoWriter(
            f"{model_name}_side_by_side.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            test_source.fps,
            (test_source.width * 2, test_source.height),
        )

    predictions = {}

    # Prediction phase
    print("Predicting...")
    masks_and_frames = zip(
        model.predict_from_source(test_source),
        enumerate(iter(test_source), start=test_source.start_frame),
    )

    for mask, (frame_id, og_frame) in tqdm(masks_and_frames, total=test_source.n_frames):
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        mask = postprocess(mask)
        bboxes = detector.detect(mask)
        bboxes = merge_bboxes(bboxes, merge_distance=40)
        predictions[frame_id] = bboxes

        if save_videos:
            if mask.dtype != np.uint8:
                mask_uint8 = (mask.astype(bool).astype(np.uint8)) * 255
            else:
                mask_uint8 = mask

            detections_frame = draw_bboxes(og_frame.copy(), bboxes, (0, 255, 0))
            if frame_id in annotations:
                detections_frame = draw_bboxes(detections_frame, annotations[frame_id], (255, 0, 0))

            mask_writer.write(mask_uint8)
            detection_writer.write(detections_frame)

            mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
            side_by_side_writer.write(np.concatenate([mask_bgr, detections_frame], axis=1))

    if save_videos:
        mask_writer.release()
        detection_writer.release()
        side_by_side_writer.release()

    # Evaluation
    gt_for_eval = {fid: boxes for fid, boxes in annotations.items() if fid in predictions}
    metrics = evaluate_detections(gt_for_eval, predictions)

    print(f"\nResults for {model_name}:")
    show_metrics(metrics)

    return metrics


def main():
    # Configuration
    ANNOTATIONS_PATH = "../ai_challenge_s03_c010-full_annotation.xml"
    VIDEO_PATH = "../AICity_data/AICity_data/train/S03/c010/vdo.avi"
    ROI_PATH = "../AICity_data/AICity_data/train/S03/c010/roi.jpg"
    SAVE_VIDEOS = True  # Set to False to skip video generation and speed up comparison

    # Setup data sources
    train_source = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
    test_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)
    annotations = load_annotations(ANNOTATIONS_PATH)

    # Setup postprocessing pipeline
    postprocess = MaskPostprocess(
        Opening((5, 5)),
        Closing((20, 20)),
        Dilate((7, 7)),
        RemoveSmallBlobs(300),
    )

    # Setup detector
    detector = TemporalCarDetector(
        base_detector=CarDetector(
            area=[1000, 10000000000],
            aspect_ratio=[0.1, 10.0],
            fill_ratio=[0.2, 1.0]
        ),
        n_frames=3,
        threshold=0.5
    )

    # Define models to compare
    models = [
        # Custom implementations
        (GrayGaussianModel(alpha=3.0), "GrayGaussian"),
        (AdaptiveGrayGaussianModel(alpha=3.0, mean_rho=0.05, variance_rho=0.05), "AdaptiveGrayGaussian"),

        # OpenCV implementations
        (MOGBackgroundSubtractor(history=200, n_mixtures=5), "MOG"),
        (MOG2BackgroundSubtractor(history=500, var_threshold=16, detect_shadows=True), "MOG2"),
        (LSBPBackgroundSubtractor(n_samples=20), "LSBP"),
    ]

    # Store results for comparison
    all_metrics = {}

    # Evaluate each model
    for model, model_name in models:
        # Create fresh sources for each model
        train_src = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
        test_src = VideoPartSource(VIDEO_PATH, 0.25, 1.0)

        try:
            metrics = evaluate_model(
                model, model_name, train_src, test_src,
                annotations, postprocess, detector, SAVE_VIDEOS
            )
            all_metrics[model_name] = metrics
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'AP':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*80)

    for model_name, metrics in all_metrics.items():
        ap = metrics.get('AP', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        print(f"{model_name:<25} {ap:<10.4f} {precision:<12.4f} {recall:<10.4f}")


if __name__ == '__main__':
    main()
