import os
import json
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_parallel_coordinate,
)

from src.bounding_box import BoundingBox
from src.trackeval_metrics import compute_trackeval_metrics, print_trackeval_metrics
from task_2_1 import (
    Detection,
    Track,
    compute_iou_matrix,
    load_annotations_with_tracks,
    save_tracking_results_mot,
    save_gt_mot_format,
    run_detector,
    visualize_tracking,
)


# ─── Kalman Filter single-object tracker ──────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single object using a Kalman filter with constant-velocity motion.

    State  : x = [u, v, s, r, u', v', s']^T   (7 x 1)
    Measurement : z = [u, v, s, r]^T           (4 x 1)
    """

    def __init__(self, bbox: BoundingBox, track_id: int):
        self.id = track_id

        # ── State transition (constant velocity) ──────────────────────────────
        # u  = u  + u'
        # v  = v  + v'
        # s  = s  + s'
        # r  = r            (constant aspect ratio)
        # u' = u', v' = v', s' = s'
        self.F = np.eye(7, dtype=float)
        self.F[0, 4] = 1.0   # u  += u'
        self.F[1, 5] = 1.0   # v  += v'
        self.F[2, 6] = 1.0   # s  += s'

        # ── Measurement matrix ────────────────────────────────────────────────
        # z = H @ x  =>  observe the first 4 state variables
        self.H = np.zeros((4, 7), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # ── Error covariance (SORT defaults) ──────────────────────────────────
        self.P = np.eye(7, dtype=float)
        self.P[4:, 4:] *= 1000.0   # high uncertainty for latent velocities
        self.P *= 10.0

        # ── Process noise ─────────────────────────────────────────────────────
        self.Q = np.eye(7, dtype=float)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6]   *= 0.01

        # ── Measurement noise ─────────────────────────────────────────────────
        self.R = np.eye(4, dtype=float)
        self.R[2:, 2:] *= 10.0

        # ── Initialise state from first detection ─────────────────────────────
        self.x = self._bbox_to_z(bbox)

        # ── Bookkeeping ───────────────────────────────────────────────────────
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

    # ── Conversion helpers ────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_z(bbox: BoundingBox) -> np.ndarray:
        """BoundingBox → Kalman measurement / initial state (7 x 1)."""
        w = bbox.right  - bbox.left
        h = bbox.bottom - bbox.top
        u = bbox.left + w / 2.0
        v = bbox.top  + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([[u], [v], [s], [r], [0.0], [0.0], [0.0]])

    def _state_to_bbox(self) -> BoundingBox:
        """Current Kalman state → BoundingBox."""
        u = float(self.x[0])
        v = float(self.x[1])
        s = max(float(self.x[2]), 1e-6)
        r = max(float(self.x[3]), 1e-6)
        w = float(np.sqrt(s * r))
        h = s / w
        return BoundingBox(
            top=v - h / 2.0,
            bottom=v + h / 2.0,
            left=u - w / 2.0,
            right=u + w / 2.0,
            confidence=None,
        )

    # ── Kalman filter steps ───────────────────────────────────────────────────

    def predict(self) -> BoundingBox:
        """Advance state one time step; return predicted bounding box."""
        # Clip s+s' to prevent negative area
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self._state_to_bbox()

    def update(self, bbox: BoundingBox):
        """Update state with a new measurement."""
        z = np.array([
            [(bbox.left   + bbox.right)  / 2.0],
            [(bbox.top    + bbox.bottom) / 2.0],
            [(bbox.right  - bbox.left) * (bbox.bottom - bbox.top)],
            [(bbox.right  - bbox.left) / max(bbox.bottom - bbox.top, 1e-6)],
        ])

        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hit_streak += 1


# ─── Multi-object Kalman Filter tracker ───────────────────────────────────────

class KalmanFilterTracker:
    """
    SORT-style multi-object tracker.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to accept a detection–track match.
    max_age : int
        Frames a track survives without any matched detection.
    min_hits : int
        Consecutive hits required before a new track is reported.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.kalman_trackers: List[KalmanBoxTracker] = []
        self.tracks: Dict[int, Track] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.frame_count = 0
        self._next_id = 1

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame_id: int, detections: List[BoundingBox]) -> List[Detection]:
        self.frame_count += 1

        # 1. Predict next position for every active tracker
        predicted: List[BoundingBox] = []
        valid_trackers: List[KalmanBoxTracker] = []
        for trk in self.kalman_trackers:
            bbox = trk.predict()
            if not any(np.isnan([bbox.left, bbox.top, bbox.right, bbox.bottom])):
                predicted.append(bbox)
                valid_trackers.append(trk)
        self.kalman_trackers = valid_trackers

        # 2. Match detections to predictions (Hungarian algorithm on IoU)
        matched_trk, matched_det, unmatched_dets = self._match(predicted, detections)

        # 3. Update matched trackers
        result: List[Detection] = []
        for trk_idx, det_idx in zip(matched_trk, matched_det):
            trk = self.kalman_trackers[trk_idx]
            trk.update(detections[det_idx])

            det = Detection(
                frame_id=frame_id,
                bbox=detections[det_idx],
                track_id=trk.id,
            )
            # Always record for offline evaluation; min_hits only gates real-time output
            self._record(trk.id, det, frame_id)
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                result.append(det)

        # 4. Initialise new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(detections[det_idx], track_id=self._next_id)
            self._next_id += 1
            self.kalman_trackers.append(trk)

            det = Detection(
                frame_id=frame_id,
                bbox=detections[det_idx],
                track_id=trk.id,
            )
            self._record(trk.id, det, frame_id)
            result.append(det)

        # 5. Prune dead trackers
        self.kalman_trackers = [
            trk for trk in self.kalman_trackers
            if trk.time_since_update <= self.max_age
        ]

        return result

    def get_all_detections(self) -> List[Detection]:
        all_dets = []
        for track in self.tracks.values():
            all_dets.extend(track.detections)
        return all_dets

    # ── Private helpers ────────────────────────────────────────────────────────

    def _record(self, track_id: int, det: Detection, frame_id: int):
        if track_id not in self.tracks:
            self.tracks[track_id] = Track(track_id=track_id)
        self.tracks[track_id].detections.append(det)
        self.track_last_seen[track_id] = frame_id

    def _match(
        self,
        predicted: List[BoundingBox],
        detections: List[BoundingBox],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return (matched_trk_idx, matched_det_idx, unmatched_det_idx)."""
        if not predicted or not detections:
            return [], [], list(range(len(detections)))

        iou_matrix = compute_iou_matrix(predicted, detections)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_trk, matched_det = [], []
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_trk.append(r)
                matched_det.append(c)
                unmatched_dets.discard(c)

        return matched_trk, matched_det, sorted(unmatched_dets)


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_kalman_experiment(
    pred_per_frame: Dict[int, List[BoundingBox]],
    gt_with_tracks: Dict,
    iou_threshold: float = 0.3,
    max_age: int = 5,
    min_hits: int = 1,
) -> Tuple[List[Detection], Dict[str, float]]:
    tracker = KalmanFilterTracker(
        iou_threshold=iou_threshold,
        max_age=max_age,
        min_hits=min_hits,
    )
    for frame_id in sorted(pred_per_frame.keys()):
        tracker.update(frame_id, pred_per_frame[frame_id])

    tracked = tracker.get_all_detections()
    metrics = compute_trackeval_metrics(gt_with_tracks, tracked)
    return tracked, metrics


# ─── Optuna search ────────────────────────────────────────────────────────────

def optuna_parameter_search(
    pred_per_frame: Dict[int, List[BoundingBox]],
    gt_with_tracks: Dict,
    output_dir: str,
    n_trials: int = 200,
    metric: str = "HOTA",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "optuna_kalman.db")

    def objective(trial):
        iou_threshold = trial.suggest_float("iou_threshold", 0.01, 0.8)
        max_age       = trial.suggest_int("max_age", 1, 50)
        min_hits      = trial.suggest_int("min_hits", 1, 5)

        _, metrics = run_kalman_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        return metrics[metric]

    # Delete stale DB if all completed trials have 0 value (broken run)
    if os.path.exists(db_path):
        try:
            _check_study = optuna.load_study(
                study_name="kalman_tracker_study",
                storage=f"sqlite:///{db_path}",
            )
            completed = [t for t in _check_study.trials if t.value is not None]
            if completed and all(t.value == 0.0 for t in completed):
                print(f"  Stale Optuna DB detected (all {len(completed)} trials = 0.0). Deleting...")
                os.remove(db_path)
        except Exception:
            pass

    study = optuna.create_study(
        study_name="kalman_tracker_study",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    print(f"\n{'='*60}")
    print("OPTUNA BAYESIAN SEARCH — Kalman Filter Tracker")
    print(f"  Trials: {n_trials}  |  Metric: {metric}")
    print(f"{'='*60}\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest {metric}: {study.best_value:.4f}")
    print(f"  iou_threshold = {best['iou_threshold']:.4f}")
    print(f"  max_age       = {best['max_age']}")
    print(f"  min_hits      = {best['min_hits']}")

    # Save visualisation plots
    plots_dir = os.path.join(output_dir, "optuna_plots")
    os.makedirs(plots_dir, exist_ok=True)
    for name, fn in [
        ("optimization_history",  plot_optimization_history),
        ("param_importances",     plot_param_importances),
        ("parallel_coordinate",   plot_parallel_coordinate),
    ]:
        try:
            fig = fn(study)
            fig.write_html(os.path.join(plots_dir, f"{name}.html"))
            fig.write_image(os.path.join(plots_dir, f"{name}.png"), scale=2)
        except Exception as e:
            print(f"  Plot {name} skipped: {e}")

    try:
        fig = plot_contour(study, params=["iou_threshold", "max_age"])
        fig.write_html(os.path.join(plots_dir, "contour.html"))
        fig.write_image(os.path.join(plots_dir, "contour.png"), scale=2)
    except Exception as e:
        print(f"  Contour plot skipped: {e}")

    # Persist results
    summary = {
        "best_params": best,
        "best_value":  study.best_value,
        "metric":      metric,
        "n_trials":    len(study.trials),
        "all_trials": [
            {"number": t.number, "params": t.params,
             "value": t.value, "user_attrs": t.user_attrs}
            for t in study.trials
        ],
    }
    with open(os.path.join(output_dir, "optuna_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _, best_metrics = run_kalman_experiment(
        pred_per_frame, gt_with_tracks,
        iou_threshold=best["iou_threshold"],
        max_age=best["max_age"],
        min_hits=best["min_hits"],
    )
    return {"best_params": best, "best_value": study.best_value,
            "best_metrics": best_metrics, "study": study}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    VIDEO_PATH = "/home/priubrogent/01_MCV/C6/00_PROJECT/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH   = "/home/priubrogent/01_MCV/C6/00_PROJECT/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"
    OUTPUT_DIR = "/home/priubrogent/01_MCV/C6/00_PROJECT/traffic_monitoring/week2/results/task_2_2"

    MODEL_NAME     = "yolov10s.pt"
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD  = 0.3
    MAX_AGE        = 5
    MIN_HITS       = 1

    DO_PARAM_SEARCH = True
    CREATE_VIS      = True
    N_TRIALS        = 300

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in (VIDEO_PATH, XML_PATH):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

    _, gt_with_tracks = load_annotations_with_tracks(XML_PATH)

    # ── Step 1: Detection ──────────────────────────────────────────────────────
    print("\n=== Step 1: Running Object Detection ===")
    pred_per_frame = run_detector(VIDEO_PATH, MODEL_NAME, CONF_THRESHOLD)
    total_dets = sum(len(v) for v in pred_per_frame.values())
    print(f"Detections in {len(pred_per_frame)} frames  ({total_dets} total boxes)")
    if not pred_per_frame:
        print("ERROR: Detector found zero car detections. Checking what classes the model detects...")
        from ultralytics import YOLO
        import cv2
        _cap = cv2.VideoCapture(VIDEO_PATH)
        ret, _frame = _cap.read()
        _cap.release()
        if ret:
            _model = YOLO(MODEL_NAME)
            _results = _model.predict(_frame, verbose=False, conf=0.1)
            _class_counts: Dict[int, int] = {}
            for _r in _results:
                for _box in _r.boxes:
                    _cid = int(_box.cls[0])
                    _class_counts[_cid] = _class_counts.get(_cid, 0) + 1
            print(f"  Classes detected in first frame (conf>0.1): {_class_counts}")
            print(f"  Model names: { {k: _model.names[k] for k in sorted(_class_counts)} }")
        return

    # ── Step 2: Hyper-parameter search (optional) ──────────────────────────────
    if DO_PARAM_SEARCH:
        search = optuna_parameter_search(
            pred_per_frame, gt_with_tracks,
            output_dir=OUTPUT_DIR,
            n_trials=N_TRIALS,
            metric="HOTA",
        )
        best = search["best_params"]
        IOU_THRESHOLD = best["iou_threshold"]
        MAX_AGE       = best["max_age"]
        MIN_HITS      = best["min_hits"]
        print(f"\nUsing best params: IoU={IOU_THRESHOLD:.4f}, "
              f"MaxAge={MAX_AGE}, MinHits={MIN_HITS}")

    # ── Step 3: Final tracking run ─────────────────────────────────────────────
    print("\n=== Step 2: Running Kalman Filter Tracker ===")
    tracked_detections, metrics = run_kalman_experiment(
        pred_per_frame, gt_with_tracks,
        iou_threshold=IOU_THRESHOLD,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
    )
    n_tracks = len(set(d.track_id for d in tracked_detections))
    print(f"Tracks: {n_tracks}  |  Detections: {len(tracked_detections)}")
    print_trackeval_metrics(metrics)

    # ── Step 4: Save outputs ───────────────────────────────────────────────────
    print("\n=== Step 3: Saving Results ===")
    pred_mot = os.path.join(OUTPUT_DIR, "pred_tracking.txt")
    save_tracking_results_mot(tracked_detections, pred_mot)
    print(f"Predictions → {pred_mot}")

    gt_mot = os.path.join(OUTPUT_DIR, "gt_tracking.txt")
    save_gt_mot_format(gt_with_tracks, gt_mot)
    print(f"Ground truth → {gt_mot}")

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"Metrics     → {metrics_path}")

    # ── Step 5: Visualise ──────────────────────────────────────────────────────
    if CREATE_VIS:
        print("\n=== Step 4: Creating Visualisation ===")
        vis_path = os.path.join(OUTPUT_DIR, "tracking_visualization.mp4")
        visualize_tracking(VIDEO_PATH, tracked_detections, vis_path)

    print("\n=== Task 2.2 Complete ===")
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
