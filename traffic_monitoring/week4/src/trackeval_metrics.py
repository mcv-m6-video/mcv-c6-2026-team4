from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import trackeval

from src.bounding_box import BoundingBox


def _iou(b1, b2):
    xl = max(b1.left, b2.left)
    yt = max(b1.top,  b2.top)
    xr = min(b1.right, b2.right)
    yb = min(b1.bottom, b2.bottom)
    if xr < xl or yb < yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    a1 = (b1.right - b1.left) * (b1.bottom - b1.top)
    a2 = (b2.right - b2.left) * (b2.bottom - b2.top)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def build_trackeval_data(gt_with_tracks, pred_detections):
    pred_by_frame = defaultdict(list)
    for det in pred_detections:
        pred_by_frame[det.frame_id].append(det)

    all_frames = sorted(set(gt_with_tracks) | set(pred_by_frame))

    gt_id_map = {}
    pred_id_map = {}

    def gt_idx(raw_id):
        if raw_id not in gt_id_map:
            gt_id_map[raw_id] = len(gt_id_map)
        return gt_id_map[raw_id]

    def pred_idx(raw_id):
        if raw_id not in pred_id_map:
            pred_id_map[raw_id] = len(pred_id_map)
        return pred_id_map[raw_id]

    gt_ids_list = []
    tracker_ids_list = []
    similarity_list = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for frame in all_frames:
        gt_entries = gt_with_tracks.get(frame, [])
        pred_entries = pred_by_frame.get(frame, [])

        gt_raw = [gt_idx(tid) for _, tid in gt_entries]
        pred_raw = [pred_idx(d.track_id) for d in pred_entries]

        gt_ids_list.append(np.array(gt_raw, dtype=int))
        tracker_ids_list.append(np.array(pred_raw, dtype=int))

        n_gt = len(gt_entries)
        n_pred = len(pred_entries)
        sim = np.zeros((n_gt, n_pred), dtype=float)
        for i, (bbox_gt, _) in enumerate(gt_entries):
            for j, det in enumerate(pred_entries):
                sim[i, j] = _iou(bbox_gt, det.bbox)
        similarity_list.append(sim)

        num_gt_dets += n_gt
        num_tracker_dets += n_pred

    return {
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": similarity_list,
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(pred_id_map),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "num_timesteps": len(all_frames),
    }


def compute_trackeval_metrics(gt_with_tracks, pred_detections):
    data = build_trackeval_data(gt_with_tracks, pred_detections)

    hota_metric = trackeval.metrics.HOTA()
    identity_metric = trackeval.metrics.Identity()
    clear_metric = trackeval.metrics.CLEAR()

    hota_res = hota_metric.eval_sequence(data)
    identity_res = identity_metric.eval_sequence(data)
    clear_res = clear_metric.eval_sequence(data)

    return {
        "HOTA":   float(np.mean(hota_res["HOTA"])),
        "DetA":   float(np.mean(hota_res["DetA"])),
        "AssA":   float(np.mean(hota_res["AssA"])),
        "IDF1":   float(identity_res["IDF1"]),
        "IDTP":   int(identity_res["IDTP"]),
        "IDFP":   int(identity_res["IDFP"]),
        "IDFN":   int(identity_res["IDFN"]),
        "MOTA":   float(clear_res["MOTA"]),
        "MOTP":   float(clear_res.get("MOTP", 0.0)),
        "FP":     int(clear_res["CLR_FP"]),
        "FN":     int(clear_res["CLR_FN"]),
        "IDS":    int(clear_res["IDSW"]),
    }


def print_trackeval_metrics(metrics):
    print("\n" + "=" * 50)
    print("TRACKING METRICS (TrackEval)")
    print("=" * 50)
    print(f"  HOTA: {metrics['HOTA']:.4f}  (DetA={metrics['DetA']:.4f}, AssA={metrics['AssA']:.4f})")
    print(f"  IDF1: {metrics['IDF1']:.4f}  (TP={metrics['IDTP']}, FP={metrics['IDFP']}, FN={metrics['IDFN']})")
    print(f"  MOTA: {metrics['MOTA']:.4f}  (MOTP={metrics['MOTP']:.4f})")
    print(f"  FP={metrics['FP']}  FN={metrics['FN']}  IDS={metrics['IDS']}")
    print("=" * 50)
