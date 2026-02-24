
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
import matplotlib.pyplot as plt


def save_confidence_examples(mask, detections, low_threshold=0.3, high_threshold=0.7):
    """Save examples of low and high confidence detections with their masks."""
    # Separate detections by confidence
    low_conf = [d for d in detections if d.confidence and d.confidence < low_threshold]
    high_conf = [d for d in detections if d.confidence and d.confidence > high_threshold]

    if not low_conf: # and not high_conf:
        return False

    mask_bgr = cv2.cvtColor(mask if mask.dtype == np.uint8 else (mask.astype(bool).astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

    if low_conf:
        # Draw low confidence detections in red
        low_vis = mask_bgr.copy()
        for det in low_conf:
            cv2.rectangle(low_vis, (int(det.left), int(det.top)), (int(det.right), int(det.bottom)), (0, 0, 255), 2)
            cv2.putText(low_vis, f"conf={det.confidence:.2f}", (int(det.left), int(det.top)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite("low_confidence_example.png", low_vis)
        print(f"Saved low_confidence_example.png with {len(low_conf)} detections")

    if False and high_conf:
        # Draw high confidence detections in green
        high_vis = mask_bgr.copy()
        for det in high_conf:
            cv2.rectangle(high_vis, (int(det.left), int(det.top)), (int(det.right), int(det.bottom)), (0, 255, 0), 2)
            cv2.putText(high_vis, f"conf={det.confidence:.2f}", (int(det.left), int(det.top)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite("high_confidence_example.png", high_vis)
        print(f"Saved high_confidence_example.png with {len(high_conf)} detections")

    return True


if __name__ == '__main__':
    ANNOTATIONS_PATH = "../ai_challenge_s03_c010-full_annotation.xml"
    VIDEO_PATH = "../AICity_data/AICity_data/train/S03/c010/vdo.avi"
    ROI_PATH = "../AICity_data/AICity_data/train/S03/c010/roi.jpg"
    OUTPUT_PATH = "predicted_masks.avi"
    
    train_source = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
    test_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)
    annotations = load_annotations(ANNOTATIONS_PATH)
    
    # model = AdaptiveGrayGaussianModel(2.84, mean_rho=0.05, variance_rho=0.05, std_bias=4.769977313430562)
    model = GrayGaussianModel(3.0, 6.5)
    
    # TODO: for the moment these are arbitrary hyperparameters
    postprocess = MaskPostprocess(
        Opening((5, 5)),
        Closing((15, 15)),
        Dilate((9, 9)),
        RemoveSmallBlobs(632),
    )
    
    detector = TemporalCarDetector(
        base_detector=CarDetector(
            area=[1550, 10000000000], # 109499 ?
            aspect_ratio=[0.22118807684047892, 4.474026431574192],
            fill_ratio=[0.20821692159573382, 1.0]
        ),
        n_frames=3,
        threshold=0.5
    )
    
    detector = CarDetector(
        area=[1550, 10000000000], # 109499 ?
        aspect_ratio=[0.22118807684047892, 4.474026431574192],
        fill_ratio=[0.20821692159573382, 1.0]
    )
    
    pipeline = DetectionPipeline(
        background_model=model,
        shadow_remover=HSVBackgroundComparison(),
        mask_posprocess=postprocess,
        detector=detector,
        bbox_merger=PolBoundingBoxMerger(merge_distance=28)
    )

    print("Fitting")
    pipeline.fit_from_source(train_source)
    
    cv2.imwrite("color_background.png", np.round(model.background).astype(np.uint8))
    cv2.imwrite("gray_background.png", np.round(model.mean).astype(np.uint8))
    cv2.imwrite("std.png", np.round(model.std * 255 / model.std.max()).astype(np.uint8))

    # Create legend images mapping intensity to std values
    std_min = np.min(model.std)
    std_max = np.max(model.std)

    # Vertical legend
    fig, ax = plt.subplots(figsize=(1.5, 6))
    fig.subplots_adjust(right=0.5)
    norm = plt.Normalize(vmin=std_min, vmax=std_max)
    cmap = plt.cm.gray
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical'
    )
    cbar.set_label('Standard Deviation', rotation=270, labelpad=20)
    plt.savefig('std_legend_vertical.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Horizontal legend
    fig, ax = plt.subplots(figsize=(6, 1))
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    cbar.set_label('Standard Deviation')
    plt.savefig('std_legend_horizontal.png', bbox_inches='tight', dpi=150)
    plt.close()

    # exit()

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
    found_examples = False  # Track if we've found good examples

    print("Predicting")
    masks_and_frames = zip(
        pipeline.predict_from_source_with_extras(test_source),
        range(test_source.start_frame, test_source.end_frame) # TODO: +1???
    )

    for (detections, extras), frame_id in tqdm(masks_and_frames, total=test_source.n_frames): # bastante F hacer esto
        predictions[frame_id] = detections

        # Try to save examples of low/high confidence detections
        if not found_examples:
            found_examples = save_confidence_examples(extras["postprocessed_mask"], detections)
        
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
