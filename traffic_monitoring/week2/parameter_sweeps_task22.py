import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from task_2_1 import (
    run_detector,
    load_annotations_with_tracks,
)
from task_2_2 import run_kalman_experiment


def iou_sweep(pred_per_frame, gt_with_tracks, output_dir, fixed_max_age=50, fixed_min_hits=1, n_values=50):
    iou_values = np.linspace(0.01, 1.0, n_values)
    results = []

    print(f"\n=== IoU Threshold Sweep (max_age={fixed_max_age}, min_hits={fixed_min_hits}) ===")
    for iou_thr in tqdm(iou_values, desc="IoU sweep"):
        _, metrics = run_kalman_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=iou_thr,
            max_age=fixed_max_age,
            min_hits=fixed_min_hits
        )
        results.append({
            "iou_threshold": float(iou_thr),
            "max_age": fixed_max_age,
            "min_hits": fixed_min_hits,
            **metrics
        })

    output_path = os.path.join(output_dir, "iou_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")

    return results


def max_age_sweep(pred_per_frame, gt_with_tracks, output_dir, fixed_iou=0.1, fixed_min_hits=1, n_values=100):
    max_age_values = list(range(1, n_values + 1))
    results = []

    print(f"\n=== Max Age Sweep (iou_threshold={fixed_iou}, min_hits={fixed_min_hits}) ===")
    for max_age in tqdm(max_age_values, desc="Max age sweep"):
        _, metrics = run_kalman_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=fixed_iou,
            max_age=max_age,
            min_hits=fixed_min_hits
        )
        results.append({
            "iou_threshold": fixed_iou,
            "max_age": int(max_age),
            "min_hits": fixed_min_hits,
            **metrics
        })

    output_path = os.path.join(output_dir, "max_age_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")

    return results


def min_hits_sweep(pred_per_frame, gt_with_tracks, output_dir, fixed_iou=0.1, fixed_max_age=50, n_values=10):
    min_hits_values = list(range(1, n_values + 1))
    results = []

    print(f"\n=== Min Hits Sweep (iou_threshold={fixed_iou}, max_age={fixed_max_age}) ===")
    for min_hits in tqdm(min_hits_values, desc="Min hits sweep"):
        _, metrics = run_kalman_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=fixed_iou,
            max_age=fixed_max_age,
            min_hits=min_hits
        )
        results.append({
            "iou_threshold": fixed_iou,
            "max_age": fixed_max_age,
            "min_hits": int(min_hits),
            **metrics
        })

    output_path = os.path.join(output_dir, "min_hits_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")

    return results


def generate_plots(output_dir, optimal_iou, optimal_max_age, optimal_min_hits):
    print("\n=== Generating Plots ===")

    with open(os.path.join(output_dir, "iou_sweep_results.json")) as f:
        iou_results = json.load(f)

    with open(os.path.join(output_dir, "max_age_sweep_results.json")) as f:
        max_age_results = json.load(f)

    with open(os.path.join(output_dir, "min_hits_sweep_results.json")) as f:
        min_hits_results = json.load(f)

    iou_values = [r["iou_threshold"] for r in iou_results]
    max_age_values = [r["max_age"] for r in max_age_results]
    min_hits_values = [r["min_hits"] for r in min_hits_results]

    metrics = ["HOTA", "IDF1", "MOTA", "AssA", "DetA"]

    # IoU sweep - all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        values = [r[metric] for r in iou_results]
        ax.plot(iou_values, values, label=metric, linewidth=2)

    ax.axvline(x=optimal_iou, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Optimal ({optimal_iou:.4f})')
    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Tracking Metrics vs IoU Threshold (Kalman Filter, max_age={optimal_max_age})", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_sweep_all_metrics.png"), dpi=150)
    print(f"Saved: iou_sweep_all_metrics.png")
    plt.close()

    # Max age sweep - all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        values = [r[metric] for r in max_age_results]
        ax.plot(max_age_values, values, label=metric, linewidth=2)

    ax.axvline(x=optimal_max_age, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Optimal ({optimal_max_age})')
    ax.set_xlabel("Max Age (frames)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Tracking Metrics vs Max Age (Kalman Filter, iou_threshold={optimal_iou:.4f})", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_age_sweep_all_metrics.png"), dpi=150)
    print(f"Saved: max_age_sweep_all_metrics.png")
    plt.close()

    # Min hits sweep - all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        values = [r[metric] for r in min_hits_results]
        ax.plot(min_hits_values, values, label=metric, linewidth=2, marker='o')

    ax.axvline(x=optimal_min_hits, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Optimal ({optimal_min_hits})')
    ax.set_xlabel("Min Hits", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Tracking Metrics vs Min Hits (Kalman Filter)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "min_hits_sweep_all_metrics.png"), dpi=150)
    print(f"Saved: min_hits_sweep_all_metrics.png")
    plt.close()

    # IoU sweep - HOTA only
    fig, ax = plt.subplots(figsize=(8, 5))
    hota_values = [r["HOTA"] for r in iou_results]
    ax.plot(iou_values, hota_values, 'b-', linewidth=2.5)
    ax.axvline(x=optimal_iou, color='r', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_iou:.4f})')
    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel("HOTA", fontsize=12)
    ax.set_title("HOTA vs IoU Threshold (Kalman Filter)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_sweep_hota.png"), dpi=150)
    print(f"Saved: iou_sweep_hota.png")
    plt.close()

    # Max age sweep - HOTA only
    fig, ax = plt.subplots(figsize=(8, 5))
    hota_values = [r["HOTA"] for r in max_age_results]
    ax.plot(max_age_values, hota_values, 'b-', linewidth=2.5)
    ax.axvline(x=optimal_max_age, color='r', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_max_age})')
    ax.set_xlabel("Max Age (frames)", fontsize=12)
    ax.set_ylabel("HOTA", fontsize=12)
    ax.set_title("HOTA vs Max Age (Kalman Filter)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_age_sweep_hota.png"), dpi=150)
    print(f"Saved: max_age_sweep_hota.png")
    plt.close()

    # Min hits sweep - HOTA only
    fig, ax = plt.subplots(figsize=(8, 5))
    hota_values = [r["HOTA"] for r in min_hits_results]
    ax.plot(min_hits_values, hota_values, 'b-', linewidth=2.5, marker='o')
    ax.axvline(x=optimal_min_hits, color='r', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_min_hits})')
    ax.set_xlabel("Min Hits", fontsize=12)
    ax.set_ylabel("HOTA", fontsize=12)
    ax.set_title("HOTA vs Min Hits (Kalman Filter)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "min_hits_sweep_hota.png"), dpi=150)
    print(f"Saved: min_hits_sweep_hota.png")
    plt.close()

    # ID switches plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ids_iou = [r["IDS"] for r in iou_results]
    axes[0].plot(iou_values, ids_iou, 'g-', linewidth=2.5)
    axes[0].axvline(x=optimal_iou, color='r', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_iou:.4f})')
    axes[0].set_xlabel("IoU Threshold", fontsize=12)
    axes[0].set_ylabel("ID Switches", fontsize=12)
    axes[0].set_title("ID Switches vs IoU Threshold", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ids_age = [r["IDS"] for r in max_age_results]
    axes[1].plot(max_age_values, ids_age, 'g-', linewidth=2.5)
    axes[1].axvline(x=optimal_max_age, color='r', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_max_age})')
    axes[1].set_xlabel("Max Age (frames)", fontsize=12)
    axes[1].set_ylabel("ID Switches", fontsize=12)
    axes[1].set_title("ID Switches vs Max Age", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    ids_min_hits = [r["IDS"] for r in min_hits_results]
    axes[2].plot(min_hits_values, ids_min_hits, 'g-', linewidth=2.5, marker='o')
    axes[2].axvline(x=optimal_min_hits, color='r', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_min_hits})')
    axes[2].set_xlabel("Min Hits", fontsize=12)
    axes[2].set_ylabel("ID Switches", fontsize=12)
    axes[2].set_title("ID Switches vs Min Hits", fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "id_switches_sweeps.png"), dpi=150)
    print(f"Saved: id_switches_sweeps.png")
    plt.close()

    # Combined 3-panel plot for slides
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # IoU
    for metric in metrics:
        values = [r[metric] for r in iou_results]
        axes[0].plot(iou_values, values, label=metric, linewidth=2)
    axes[0].axvline(x=optimal_iou, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel("IoU Threshold", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("Metrics vs IoU Threshold", fontsize=14)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Max Age
    for metric in metrics:
        values = [r[metric] for r in max_age_results]
        axes[1].plot(max_age_values, values, label=metric, linewidth=2)
    axes[1].axvline(x=optimal_max_age, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel("Max Age (frames)", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Metrics vs Max Age", fontsize=14)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)

    # Min Hits
    for metric in metrics:
        values = [r[metric] for r in min_hits_results]
        axes[2].plot(min_hits_values, values, label=metric, linewidth=2, marker='o')
    axes[2].axvline(x=optimal_min_hits, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel("Min Hits", fontsize=12)
    axes[2].set_ylabel("Score", fontsize=12)
    axes[2].set_title("Metrics vs Min Hits", fontsize=14)
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_sweeps_combined.png"), dpi=150)
    print(f"Saved: all_sweeps_combined.png")
    plt.close()

    print("\nAll plots generated!")


def main():
    VIDEO_PATH = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"
    OUTPUT_DIR = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_2"

    MODEL_NAME = "yolov10s.pt"
    CONF_THRESHOLD = 0.5
    VEHICLE_CLASS_IDS = [2, 5, 7]

    # Load optimal params from optuna results if available
    optuna_path = os.path.join(OUTPUT_DIR, "optuna_results.json")
    if os.path.exists(optuna_path):
        with open(optuna_path) as f:
            optuna_results = json.load(f)
        OPTIMAL_IOU = optuna_results["best_params"]["iou_threshold"]
        OPTIMAL_MAX_AGE = optuna_results["best_params"]["max_age"]
        OPTIMAL_MIN_HITS = optuna_results["best_params"]["min_hits"]
        print(f"Loaded optimal params from Optuna: IoU={OPTIMAL_IOU:.4f}, max_age={OPTIMAL_MAX_AGE}, min_hits={OPTIMAL_MIN_HITS}")
    else:
        OPTIMAL_IOU = 0.1
        OPTIMAL_MAX_AGE = 50
        OPTIMAL_MIN_HITS = 1
        print(f"Using default params: IoU={OPTIMAL_IOU}, max_age={OPTIMAL_MAX_AGE}, min_hits={OPTIMAL_MIN_HITS}")

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

    # Run sweeps
    iou_results = iou_sweep(
        pred_per_frame, gt_with_tracks, OUTPUT_DIR,
        fixed_max_age=OPTIMAL_MAX_AGE,
        fixed_min_hits=OPTIMAL_MIN_HITS,
        n_values=50
    )

    max_age_results = max_age_sweep(
        pred_per_frame, gt_with_tracks, OUTPUT_DIR,
        fixed_iou=OPTIMAL_IOU,
        fixed_min_hits=OPTIMAL_MIN_HITS,
        n_values=100
    )

    min_hits_results = min_hits_sweep(
        pred_per_frame, gt_with_tracks, OUTPUT_DIR,
        fixed_iou=OPTIMAL_IOU,
        fixed_max_age=OPTIMAL_MAX_AGE,
        n_values=10
    )

    # Generate plots
    generate_plots(OUTPUT_DIR, OPTIMAL_IOU, OPTIMAL_MAX_AGE, OPTIMAL_MIN_HITS)

    print("\n=== Sweep Complete ===")
    print(f"IoU sweep: {len(iou_results)} values")
    print(f"Max age sweep: {len(max_age_results)} values")
    print(f"Min hits sweep: {len(min_hits_results)} values")
    print(f"Results and plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
