"""
    Adaptive Gaussian Model Ro Study

    Using Optuna best parameters as baseline:
    - alpha: 2.0933154680482495
    - mean_rho (best): 0.009115651856561206
    - variance_rho (best): 0.07520599715083304
    - std_bias: 4.207707917671822
    - HSV shadow removal parameters from Optuna

    To run individual ro studies in parallel:

    # Study mean_rho only (variance_rho fixed at Optuna best: 0.0752)
    python adaptive_ro_study.py --study mean_rho --ro_start 0.001 --ro_end 0.005 --ro_step 0.001 --output mean_rho_1.json &
    python adaptive_ro_study.py --study mean_rho --ro_start 0.006 --ro_end 0.020 --ro_step 0.002 --output mean_rho_2.json &

    # Study variance_rho only (mean_rho fixed at Optuna best: 0.0091)
    python adaptive_ro_study.py --study variance_rho --ro_start 0.01 --ro_end 0.10 --ro_step 0.01 --output variance_rho_1.json &
    python adaptive_ro_study.py --study variance_rho --ro_start 0.15 --ro_end 0.60 --ro_step 0.05 --output variance_rho_2.json &

    # Study combinations
    python adaptive_ro_study.py --study combination --mean_rho_values 0.005,0.009,0.015 --variance_rho_values 0.05,0.075,0.10 --output combo_1.json &
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm


def format_time(seconds):
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
from src.models.adaptive_models import AdaptiveGrayGaussianModel
from src.bbox_merging import PolBoundingBoxMerger
from src.shadow_removal import HSVBackgroundComparison
from src.object_detection import CarDetector
from src.mask_postprocessing import MaskPostprocess, Opening, Closing, Dilate, RemoveSmallBlobs
from src.evaluation import evaluate_detections, load_annotations
from src.video_source import VideoPartSource
from src.pipeline import DetectionPipeline


def evaluate_single_frame(gt_boxes, pred_boxes, frame_id):
    if not gt_boxes and not pred_boxes:
        return None

    gt_dict = {frame_id: gt_boxes} if gt_boxes else {}
    pred_dict = {frame_id: pred_boxes} if pred_boxes else {}

    # Suppress stdout during per-frame evaluation
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = evaluate_detections(gt_dict, pred_dict)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return result


def run_experiment(mean_rho, variance_rho, annotations, train_source, test_source):
    # Optuna best parameters as baseline
    model = AdaptiveGrayGaussianModel(
        alpha=2.0933154680482495,
        mean_rho=mean_rho,
        variance_rho=variance_rho,
        std_bias=4.207707917671822
    )

    postprocess = MaskPostprocess(
        Opening((5, 5)),
        Closing((15, 15)),
        Dilate((9, 9)),
        RemoveSmallBlobs(632),
    )

    detector = CarDetector(
        area=[1550, 10000000000],
        aspect_ratio=[0.22118807684047892, 4.474026431574192],
        fill_ratio=[0.20821692159573382, 1.0]
    )

    pipeline = DetectionPipeline(
        background_model=model,
        shadow_remover=HSVBackgroundComparison(
            hue_threshold=29.905532124564267,
            value_ratio_min=0.5078298592725248,
            value_ratio_max=0.9948932249196348,
            saturation_diff_threshold=37.1876589771072
        ),
        mask_posprocess=postprocess,
        detector=detector,
        bbox_merger=PolBoundingBoxMerger(merge_distance=28)
    )

    pipeline.fit_from_source(train_source)

    predictions = {}
    per_frame_metrics = {}

    total_frames = test_source.end_frame - test_source.start_frame
    masks_and_frames = zip(
        pipeline.predict_from_source_with_extras(test_source),
        range(test_source.start_frame, test_source.end_frame)
    )

    for (detections, extras), frame_id in tqdm(masks_and_frames, total=total_frames, desc="Processing frames"):
        predictions[frame_id] = detections

        if frame_id in annotations:
            frame_metrics = evaluate_single_frame(
                annotations[frame_id],
                detections,
                frame_id
            )
            if frame_metrics is not None:
                per_frame_metrics[str(frame_id)] = frame_metrics

    gt_for_eval = {fid: boxes for fid, boxes in annotations.items() if fid in predictions}
    total_metrics = evaluate_detections(gt_for_eval, predictions)

    return {
        "total_metrics": total_metrics,
        "per_frame_metrics": per_frame_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive Gaussian Model Ro Study")
    parser.add_argument("--study", type=str, required=True,
                        choices=["mean_rho", "variance_rho", "combination"],
                        help="Which ro parameter(s) to study")

    # For individual ro studies
    parser.add_argument("--ro_start", type=float, help="Start value for ro")
    parser.add_argument("--ro_end", type=float, help="End value for ro")
    parser.add_argument("--ro_step", type=float, help="Step size for ro")

    # For combination study
    parser.add_argument("--mean_rho_values", type=str,
                        help="Comma-separated mean_rho values for combination study")
    parser.add_argument("--variance_rho_values", type=str,
                        help="Comma-separated variance_rho values for combination study")

    # Fixed values when studying one ro (Optuna best parameters)
    parser.add_argument("--fixed_mean_rho", type=float, default=0.009115651856561206,
                        help="Fixed mean_rho when studying variance_rho (Optuna best)")
    parser.add_argument("--fixed_variance_rho", type=float, default=0.07520599715083304,
                        help="Fixed variance_rho when studying mean_rho (Optuna best)")

    # Common arguments
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--annotations", type=str,
                        default="../ai_challenge_s03_c010-full_annotation.xml",
                        help="Path to annotations XML")
    parser.add_argument("--video", type=str,
                        default="../AICity_data/AICity_data/train/S03/c010/vdo.avi",
                        help="Path to video")

    args = parser.parse_args()

    annotations = load_annotations(args.annotations)

    results = []

    if args.study == "mean_rho":
        # Study mean_rho, keep variance_rho fixed
        mean_rho_values = np.arange(args.ro_start, args.ro_end + args.ro_step/2, args.ro_step)
        print(f"Studying mean_rho: {len(mean_rho_values)} values from {args.ro_start} to {args.ro_end}")
        print(f"Fixed variance_rho = {args.fixed_variance_rho} (Optuna best)")
        print(f"Fixed alpha = 2.0933154680482495, std_bias = 4.207707917671822 (Optuna best)")

        for i, mean_rho in enumerate(mean_rho_values):
            print(f"\n[{i+1}/{len(mean_rho_values)}] Testing mean_rho = {mean_rho:.4f}, variance_rho = {args.fixed_variance_rho}")

            # Create fresh sources for each experiment (adaptive model modifies internal state)
            train_source = VideoPartSource(args.video, 0.0, 0.25)
            test_source = VideoPartSource(args.video, 0.25, 1.0)
            experiment_results = run_experiment(
                mean_rho, args.fixed_variance_rho,
                annotations, train_source, test_source
            )

            results.append({
                "mean_rho": float(mean_rho),
                "variance_rho": float(args.fixed_variance_rho),
                "total_metrics": experiment_results["total_metrics"],
                "per_frame_metrics": experiment_results["per_frame_metrics"]
            })

            print(f"AP = {experiment_results['total_metrics']['AP']:.4f}, "
                  f"AP50 = {experiment_results['total_metrics']['AP50']:.4f}")

    elif args.study == "variance_rho":
        # Study variance_rho, keep mean_rho fixed
        variance_rho_values = np.arange(args.ro_start, args.ro_end + args.ro_step/2, args.ro_step)
        print(f"Studying variance_rho: {len(variance_rho_values)} values from {args.ro_start} to {args.ro_end}")
        print(f"Fixed mean_rho = {args.fixed_mean_rho} (Optuna best)")
        print(f"Fixed alpha = 2.0933154680482495, std_bias = 4.207707917671822 (Optuna best)")

        for i, variance_rho in enumerate(variance_rho_values):
            print(f"\n[{i+1}/{len(variance_rho_values)}] Testing mean_rho = {args.fixed_mean_rho}, variance_rho = {variance_rho:.4f}")

            # Create fresh sources for each experiment (adaptive model modifies internal state)
            train_source = VideoPartSource(args.video, 0.0, 0.25)
            test_source = VideoPartSource(args.video, 0.25, 1.0)
            experiment_results = run_experiment(
                args.fixed_mean_rho, variance_rho,
                annotations, train_source, test_source
            )

            results.append({
                "mean_rho": float(args.fixed_mean_rho),
                "variance_rho": float(variance_rho),
                "total_metrics": experiment_results["total_metrics"],
                "per_frame_metrics": experiment_results["per_frame_metrics"]
            })

            print(f"AP = {experiment_results['total_metrics']['AP']:.4f}, "
                  f"AP50 = {experiment_results['total_metrics']['AP50']:.4f}")

    elif args.study == "combination":
        # Study all combinations
        mean_rho_values = [float(x) for x in args.mean_rho_values.split(',')]
        variance_rho_values = [float(x) for x in args.variance_rho_values.split(',')]

        print(f"Studying combinations of mean_rho and variance_rho")
        print(f"mean_rho values: {mean_rho_values}")
        print(f"variance_rho values: {variance_rho_values}")
        print(f"Total combinations: {len(mean_rho_values) * len(variance_rho_values)}")
        print(f"Fixed alpha = 2.0933154680482495, std_bias = 4.207707917671822 (Optuna best)")

        combinations = [(m, v) for m in mean_rho_values for v in variance_rho_values]

        for i, (mean_rho, variance_rho) in enumerate(combinations):
            print(f"\n[{i+1}/{len(combinations)}] Testing mean_rho = {mean_rho:.4f}, variance_rho = {variance_rho:.4f}")

            # Create fresh sources for each experiment (adaptive model modifies internal state)
            train_source = VideoPartSource(args.video, 0.0, 0.25)
            test_source = VideoPartSource(args.video, 0.25, 1.0)
            experiment_results = run_experiment(
                mean_rho, variance_rho,
                annotations, train_source, test_source
            )

            results.append({
                "mean_rho": float(mean_rho),
                "variance_rho": float(variance_rho),
                "total_metrics": experiment_results["total_metrics"],
                "per_frame_metrics": experiment_results["per_frame_metrics"]
            })

            print(f"AP = {experiment_results['total_metrics']['AP']:.4f}, "
                  f"AP50 = {experiment_results['total_metrics']['AP50']:.4f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"Tested {len(results)} configurations")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
