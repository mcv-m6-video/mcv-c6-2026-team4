import os
import json
import pickle
import numpy as np
from tqdm import tqdm

from task_2_1 import (
    run_detector,
    run_tracking_experiment,
    load_annotations_with_tracks,
)


def iou_sweep(pred_per_frame, gt_with_tracks, output_dir, fixed_max_age=50, n_values=50):
    iou_values = np.linspace(0.01, 1.0, n_values)
    results = []

    print(f"\n=== IoU Threshold Sweep (max_age={fixed_max_age}) ===")
    for iou_thr in tqdm(iou_values, desc="IoU sweep"):
        _, metrics = run_tracking_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=iou_thr, max_age=fixed_max_age
        )
        results.append({
            "iou_threshold": float(iou_thr),
            "max_age": fixed_max_age,
            **metrics
        })

    output_path = os.path.join(output_dir, "iou_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")

    return results


def max_age_sweep(pred_per_frame, gt_with_tracks, output_dir, fixed_iou=0.1094, n_values=100):
    max_age_values = list(range(1, n_values + 1))
    results = []

    print(f"\n=== Max Age Sweep (iou_threshold={fixed_iou}) ===")
    for max_age in tqdm(max_age_values, desc="Max age sweep"):
        _, metrics = run_tracking_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=fixed_iou, max_age=max_age
        )
        results.append({
            "iou_threshold": fixed_iou,
            "max_age": int(max_age),
            **metrics
        })

    output_path = os.path.join(output_dir, "max_age_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")

    return results


def main():
    VIDEO_PATH = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"
    OUTPUT_DIR = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_1"

    MODEL_NAME = "yolov10s.pt"
    CONF_THRESHOLD = 0.5
    VEHICLE_CLASS_IDS = [2, 5, 7]

    FIXED_IOU = 0.1094
    FIXED_MAX_AGE = 50

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_pkl_path = os.path.join(OUTPUT_DIR, "pred_per_frame_yolov10s.pkl")

    if os.path.exists(pred_pkl_path):
        print(f"Loading cached detections from {pred_pkl_path}")
        with open(pred_pkl_path, "rb") as f:
            pred_per_frame = pickle.load(f)
    else:
        print("Running YOLOv10s detection...")
        pred_per_frame = run_detector(VIDEO_PATH, MODEL_NAME, CONF_THRESHOLD, VEHICLE_CLASS_IDS)
        with open(pred_pkl_path, "wb") as f:
            pickle.dump(pred_per_frame, f)
        print(f"Saved detections to {pred_pkl_path}")

    print(f"Detected objects in {len(pred_per_frame)} frames")

    print("Loading ground truth...")
    _, gt_with_tracks = load_annotations_with_tracks(XML_PATH)

    iou_results = iou_sweep(pred_per_frame, gt_with_tracks, OUTPUT_DIR,
                            fixed_max_age=FIXED_MAX_AGE, n_values=50)

    max_age_results = max_age_sweep(pred_per_frame, gt_with_tracks, OUTPUT_DIR,
                                     fixed_iou=FIXED_IOU, n_values=100)

    print("\n=== Sweep Complete ===")
    print(f"IoU sweep: {len(iou_results)} values")
    print(f"Max age sweep: {len(max_age_results)} values")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
