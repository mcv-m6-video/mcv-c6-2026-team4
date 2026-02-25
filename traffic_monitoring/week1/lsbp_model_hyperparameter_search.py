import argparse
import xml.etree.ElementTree as ET
from typing import Dict

import optuna
from optuna.storages import RDBStorage

from src.evaluation import evaluate_detections
from src.mask_postprocessing import Closing, Dilate, MaskPostprocess, Opening, RemoveSmallBlobs
from src.models.opencv_models import LSBPBackgroundSubtractor
from src.object_detection import BoundingBox, CarDetector, TemporalCarDetector, merge_bboxes
from src.video_source import VideoPartSource


def load_annotations(path: str) -> Dict[int, list[BoundingBox]]:
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
            bbox = BoundingBox(top=ytl, bottom=ybr, left=xtl, right=xbr, confidence=1.0)
            if frame not in boxes_per_frame:
                boxes_per_frame[frame] = [bbox]
            else:
                boxes_per_frame[frame].append(bbox)

    return boxes_per_frame


def objective(trial: optuna.Trial, annotations: Dict[int, list[BoundingBox]],
              train_source: VideoPartSource, test_source: VideoPartSource,
              use_temporal: bool) -> float:

    # Model hyperparameters (structural params only)
    # motion_compensation fixed to 0 (no compensation for static camera)
    # t_inc, t_dec, r_scale, r_inc_dec fixed at OpenCV defaults (control adaptive threshold
    # dynamics — low impact on a static traffic scene)
    n_samples = trial.suggest_int("n_samples", 10, 40)
    lsbp_radius = trial.suggest_int("lsbp_radius", 4, 32)
    tlsbp_threshold = trial.suggest_float("tlsbp_threshold", 0.5, 5.0)
    lsbp_threshold = trial.suggest_int("lsbp_threshold", 4, 16)
    min_count = trial.suggest_int("min_count", 1, 5)

    # Fixed postprocessing (best-known values from still model study)
    opening_size = 5
    closing_size = 15
    dilate_size = 9
    remove_small_blobs = 632

    # Fixed detector (best-known values from still model study)
    area_min = 1550
    area_max = 10000000000
    aspect_ratio_min = 0.22118807684047892
    aspect_ratio_max = 4.474026431574192
    fill_ratio_min = 0.20821692159573382
    fill_ratio_max = 1.0

    # Fixed temporal (best-known values from still model study)
    if use_temporal:
        n_frames = 3
        temporal_threshold = 0.5

    # Fixed BBOX postprocessing (best-known values from still model study)
    merge_distance = 28

    model = LSBPBackgroundSubtractor(
        motion_compensation=0,
        n_samples=n_samples,
        lsbp_radius=lsbp_radius,
        tlsbp_threshold=tlsbp_threshold,
        lsbp_threshold=lsbp_threshold,
        min_count=min_count,
    )
    model.fit_from_source(train_source)

    postprocess = MaskPostprocess(
        Opening((opening_size, opening_size)),
        Closing((closing_size, closing_size)),
        Dilate((dilate_size, dilate_size)),
        RemoveSmallBlobs(remove_small_blobs),
    )

    base_detector = CarDetector(
        area=[area_min, area_max],
        aspect_ratio=[aspect_ratio_min, aspect_ratio_max],
        fill_ratio=[fill_ratio_min, fill_ratio_max],
    )

    if use_temporal:
        detector = TemporalCarDetector(base_detector, n_frames=n_frames, threshold=temporal_threshold)
    else:
        detector = base_detector

    predictions = {}
    for mask, frame_id in zip(
        model.predict_from_source(test_source),
        range(test_source.start_frame, test_source.start_frame + test_source.n_frames)
    ):
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        processed_mask = postprocess(mask)
        bboxes = detector.detect(processed_mask)
        bboxes = merge_bboxes(bboxes, merge_distance=merge_distance)
        predictions[frame_id] = bboxes

    gt_for_eval = {fid: boxes for fid, boxes in annotations.items() if fid in predictions}

    metrics = evaluate_detections(gt_for_eval, predictions)
    ap50 = metrics["AP50"]

    return ap50


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for LSBP model")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--study-name", type=str, default="lsbp_model_optimization",
                       help="Name of the Optuna study")
    parser.add_argument("--db-path", type=str, default="optuna_lsbp_model.db",
                       help="Path to SQLite database for study storage")
    parser.add_argument("--use-temporal", action="store_true",
                       help="Use temporal detector instead of base detector")
    parser.add_argument("--annotations", type=str,
                       default="../ai_challenge_s03_c010-full_annotation.xml",
                       help="Path to annotations file")
    parser.add_argument("--video", type=str,
                       default="../AICity_data/AICity_data/train/S03/c010/vdo.avi",
                       help="Path to video file")
    args = parser.parse_args()

    print("Loading annotations and video...")
    annotations = load_annotations(args.annotations)
    train_source = VideoPartSource(args.video, 0.0, 0.25)
    test_source = VideoPartSource(args.video, 0.25, 1.0)

    storage = RDBStorage(f"sqlite:///{args.db_path}")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # we are maximizing AP50!!!
        load_if_exists=True,  # allow parallel processes to share the same database
    )

    print(f"Starting optimization for study '{args.study_name}'")
    print(f"Using temporal detector: {args.use_temporal}")
    print(f"Database: {args.db_path}")
    print(f"Number of trials: {args.n_trials}")
    print("\nTo run in parallel, execute this same command in multiple terminals.")
    print("-" * 80)

    study.optimize(
        lambda trial: objective(trial, annotations, train_source, test_source, args.use_temporal),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest AP50: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nTotal trials completed: {len(study.trials)}")
    print(f"Results saved to: {args.db_path}")
    print("\nTo visualize results, use optuna-dashboard:")
    print(f"  optuna-dashboard sqlite:///{args.db_path}")


if __name__ == "__main__":
    main()
