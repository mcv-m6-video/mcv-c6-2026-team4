"""
    To run this script:
    
    python alpha_study.py --alpha_start 0.5 --alpha_end 1.75 --alpha_step 0.25 --output results_1.json &
    python alpha_study.py --alpha_start 2.0 --alpha_end 3.25 --alpha_step 0.25 --output results_2.json &
    python alpha_study.py --alpha_start 3.5 --alpha_end 4.75 --alpha_step 0.25 --output results_3.json &
    python alpha_study.py --alpha_start 5.0 --alpha_end 5.0 --alpha_step 0.25 --output results_4.json &
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from src.models.still_models import GrayGaussianModel
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

    return evaluate_detections(gt_dict, pred_dict)


def run_alpha_experiment(alpha_value, annotations, train_source, test_source):
    model = GrayGaussianModel(alpha=alpha_value, std_bias=2.0)

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
        shadow_remover=HSVBackgroundComparison(),
        mask_posprocess=postprocess,
        detector=detector,
        bbox_merger=PolBoundingBoxMerger(merge_distance=28)
    )

    pipeline.fit_from_source(train_source)

    predictions = {}
    per_frame_metrics = {}

    masks_and_frames = zip(
        pipeline.predict_from_source_with_extras(test_source),
        range(test_source.start_frame, test_source.end_frame)
    )

    for (detections, extras), frame_id in masks_and_frames:
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
    parser = argparse.ArgumentParser(description="Alpha parameter study for GrayGaussianModel")
    parser.add_argument("--alpha_start", type=float, default=0.5, help="Start value for alpha")
    parser.add_argument("--alpha_end", type=float, default=5.0, help="End value for alpha")
    parser.add_argument("--alpha_step", type=float, default=0.5, help="Step size for alpha")
    parser.add_argument("--output", type=str, default="alpha_study_results.json", help="Output JSON file")
    parser.add_argument("--annotations", type=str,
                        default="../ai_challenge_s03_c010-full_annotation.xml",
                        help="Path to annotations XML")
    parser.add_argument("--video", type=str,
                        default="../AICity_data/AICity_data/train/S03/c010/vdo.avi",
                        help="Path to video")

    args = parser.parse_args()

    alpha_values = np.arange(args.alpha_start, args.alpha_end + args.alpha_step/2, args.alpha_step)

    print(f"Testing {len(alpha_values)} alpha values: {alpha_values}")
    print(f"Fixed std_bias = 2.0")
    print(f"Output will be saved to: {args.output}")

    train_source = VideoPartSource(args.video, 0.0, 0.25)
    annotations = load_annotations(args.annotations)

    results = []

    for alpha in tqdm(alpha_values, desc="Testing alpha values"):
        print(f"\n--- Testing alpha = {alpha:.2f} ---")


        test_source = VideoPartSource(args.video, 0.25, 1.0)


        experiment_results = run_alpha_experiment(alpha, annotations, train_source, test_source)


        results.append({
            "alpha": float(alpha),
            "total_metrics": experiment_results["total_metrics"],
            "per_frame_metrics": experiment_results["per_frame_metrics"]
        })


        print(f"Alpha {alpha:.2f}: AP = {experiment_results['total_metrics']['AP']:.4f}, "
              f"AP50 = {experiment_results['total_metrics']['AP50']:.4f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Tested {len(results)} alpha values")


if __name__ == '__main__':
    main()
