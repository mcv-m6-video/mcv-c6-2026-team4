"""
ReID network diagnostic.

For every GT vehicle that appears in 2+ cameras, extracts crops from each
camera it is seen in, runs them through the ReID network, and computes:

  - Same-vehicle, cross-camera cosine similarity  (should be HIGH → low cost)
  - Different-vehicle, cross-camera cosine similarity (should be LOW → high cost)

Reports distribution statistics, rank-1 retrieval accuracy, and saves a
histogram plot to <output_dir>/reid_similarity_distributions.png.

Usage
-----
python reid_diagnostic.py
python reid_diagnostic.py --seq S01 --cameras c001 c002 c003 c004 c005
python reid_diagnostic.py --n-crops 10 --n-diff-pairs 5000
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from src.eval import readData
from src.model import ft_net

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_ROOT    = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
SEQ_ID       = "S01"
CAMERAS      = ["c001", "c002", "c003", "c004", "c005"]
GT_PATH      = f"{DATA_ROOT}/eval/ground_truth_train.txt"
REID_WEIGHTS = "./src/net_19.pth"
OUTPUT_DIR   = Path("output/reid_diagnostic")

N_CROPS_PER_TRACK = 5    # crops sampled per vehicle per camera
N_DIFF_PAIRS      = 3000  # random different-vehicle cross-camera pairs to evaluate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReID network diagnostic")
    p.add_argument("--data-root",       default=DATA_ROOT)
    p.add_argument("--seq",             default=SEQ_ID)
    p.add_argument("--cameras",         nargs="+", default=CAMERAS)
    p.add_argument("--gt-path",         default=GT_PATH)
    p.add_argument("--reid-weights",    default=REID_WEIGHTS)
    p.add_argument("--n-crops",         type=int, default=N_CROPS_PER_TRACK,
                   help="Crops sampled per vehicle per camera")
    p.add_argument("--n-diff-pairs",    type=int, default=N_DIFF_PAIRS,
                   help="Random different-vehicle pairs to evaluate")
    p.add_argument("--output-dir",      default=str(OUTPUT_DIR))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_model_and_transform(weights_path: str, device: torch.device):
    model = ft_net(class_num=34071, ibn=True, linear_num=2048, circle=True)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model = model.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform


@torch.no_grad()
def extract_feature(model, transform, crop_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Extract a single L2-normalised ReID feature from a BGR crop."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb).unsqueeze(0).to(device)
    _, feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# GT loading and crop collection
# ---------------------------------------------------------------------------

def load_multicam_gt(gt_path: str, camera_ids: list[str]) -> dict[int, dict[str, list[tuple]]]:
    """
    Returns { vehicle_id: { cam_id: [(frame_id, x, y, w, h), ...] } }
    filtered to vehicles that appear in ≥ 2 of the requested cameras.
    """
    df = readData(gt_path)

    # Map integer camera IDs to string IDs ("c001", etc.)
    cam_int_to_str = {int(c.lstrip("c")): c for c in camera_ids}
    df = df[df["CameraId"].isin(cam_int_to_str.keys())]

    # Build lookup
    data: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples(index=False):
        cam_str = cam_int_to_str[row.CameraId]
        data[row.Id][cam_str].append((row.FrameId, row.X, row.Y, row.Width, row.Height))

    # Keep only multi-camera vehicles
    return {
        vid: dict(cam_data)
        for vid, cam_data in data.items()
        if len(cam_data) >= 2
    }


def collect_crops_for_camera(
    video_path: str,
    vehicle_frames: dict[int, list[tuple]],
    n_crops: int,
) -> dict[int, list[np.ndarray]]:
    """
    Reads `video_path` once sequentially, collecting crops for each vehicle.

    `vehicle_frames`: { vehicle_id: [(frame_id, x, y, w, h), ...] }
    Returns: { vehicle_id: [crop_bgr, ...] }  (up to n_crops per vehicle)
    """
    # Build a frame→vehicles lookup and determine which frames we need.
    frame_to_vehicles: dict[int, list[tuple]] = defaultdict(list)
    for vid, obs_list in vehicle_frames.items():
        # Evenly sample up to n_crops observations
        indices = np.linspace(0, len(obs_list) - 1, min(len(obs_list), n_crops), dtype=int)
        for i in indices:
            frame_id, x, y, w, h = obs_list[i]
            frame_to_vehicles[frame_id].append((vid, x, y, w, h))

    needed_frames = set(frame_to_vehicles.keys())
    crops: dict[int, list[np.ndarray]] = defaultdict(list)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return crops

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while frame_idx < total and needed_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1                        # GT frame IDs are 1-based

        if frame_idx not in needed_frames:
            continue
        needed_frames.discard(frame_idx)

        h_frame, w_frame = frame.shape[:2]
        for vid, x, y, w, h in frame_to_vehicles[frame_idx]:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w_frame, int(x + w))
            y2 = min(h_frame, int(y + h))
            if x2 > x1 and y2 > y1:
                crops[vid].append(frame[y1:y2, x1:x2].copy())

    cap.release()
    return crops


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ReID model …")
    model, transform = build_model_and_transform(args.reid_weights, device)

    print("Loading GT …")
    gt = load_multicam_gt(args.gt_path, args.cameras)
    print(f"  Multi-camera vehicles: {len(gt)}")

    # ------------------------------------------------------------------
    # Per-camera crop collection + feature extraction
    # ------------------------------------------------------------------
    # features[vehicle_id][cam_id] = mean L2-normalised feature vector
    features: dict[int, dict[str, np.ndarray]] = defaultdict(dict)

    for cam_id in args.cameras:
        video_path = f"{args.data_root}/train/{args.seq}/{cam_id}/vdo.avi"

        # Gather observations for this camera
        cam_vehicles = {
            vid: cam_data[cam_id]
            for vid, cam_data in gt.items()
            if cam_id in cam_data
        }
        if not cam_vehicles:
            continue

        print(f"\nCamera {cam_id}: {len(cam_vehicles)} vehicles …")
        crops_by_vehicle = collect_crops_for_camera(video_path, cam_vehicles, args.n_crops)

        for vid, crop_list in tqdm(crops_by_vehicle.items(), desc=f"  {cam_id} features", leave=False):
            if not crop_list:
                continue
            feats = []
            for crop in crop_list:
                if crop.size == 0:
                    continue
                feats.append(extract_feature(model, transform, crop, device))
            if feats:
                mean_feat = np.mean(feats, axis=0)
                norm = np.linalg.norm(mean_feat)
                features[vid][cam_id] = mean_feat / norm if norm > 0 else mean_feat

    # ------------------------------------------------------------------
    # Compute same-vehicle cross-camera similarities
    # ------------------------------------------------------------------
    same_sims: list[float] = []     # one entry per (vehicle, cam_pair) combo
    same_meta: list[tuple]  = []    # (vehicle_id, cam_i, cam_j) for inspection

    for vid, cam_feats in features.items():
        cams = list(cam_feats.keys())
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                sim = float(np.dot(cam_feats[cams[i]], cam_feats[cams[j]]))
                same_sims.append(sim)
                same_meta.append((vid, cams[i], cams[j]))

    # ------------------------------------------------------------------
    # Compute different-vehicle cross-camera similarities
    # ------------------------------------------------------------------
    vehicle_ids   = list(features.keys())
    diff_sims: list[float] = []

    rng = random.Random(42)
    attempts = 0
    while len(diff_sims) < args.n_diff_pairs and attempts < args.n_diff_pairs * 10:
        attempts += 1
        vid_a, vid_b = rng.sample(vehicle_ids, 2)
        # Pick a random camera for each vehicle (different vehicles, any cameras)
        cams_a = list(features[vid_a].keys())
        cams_b = list(features[vid_b].keys())
        cam_a = rng.choice(cams_a)
        cam_b = rng.choice(cams_b)
        sim = float(np.dot(features[vid_a][cam_a], features[vid_b][cam_b]))
        diff_sims.append(sim)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    same_arr = np.array(same_sims)
    diff_arr = np.array(diff_sims)

    print("\n" + "="*60)
    print("REID DIAGNOSTIC RESULTS")
    print("="*60)
    print(f"\nSame-vehicle, cross-camera ({len(same_arr)} pairs):")
    print(f"  mean  = {same_arr.mean():.4f}")
    print(f"  std   = {same_arr.std():.4f}")
    print(f"  min   = {same_arr.min():.4f}  max = {same_arr.max():.4f}")
    print(f"  p25   = {np.percentile(same_arr, 25):.4f}  "
          f"median = {np.median(same_arr):.4f}  "
          f"p75 = {np.percentile(same_arr, 75):.4f}")

    print(f"\nDifferent-vehicle, cross-camera ({len(diff_arr)} pairs):")
    print(f"  mean  = {diff_arr.mean():.4f}")
    print(f"  std   = {diff_arr.std():.4f}")
    print(f"  min   = {diff_arr.min():.4f}  max = {diff_arr.max():.4f}")
    print(f"  p25   = {np.percentile(diff_arr, 25):.4f}  "
          f"median = {np.median(diff_arr):.4f}  "
          f"p75 = {np.percentile(diff_arr, 75):.4f}")

    separation = same_arr.mean() - diff_arr.mean()
    print(f"\nSeparation (same_mean − diff_mean) = {separation:.4f}")
    print(f"  → {'Good separation' if separation > 0.15 else 'Poor separation — ReID is struggling'}")

    # ------------------------------------------------------------------
    # Rank-1 accuracy
    # ------------------------------------------------------------------
    # For each same-vehicle cross-camera pair, check whether this pair
    # has higher similarity than all different-vehicle pairs for the same
    # query vehicle/camera.
    rank1_correct = 0
    rank1_total   = 0

    for (vid, cam_i, cam_j), sim_pos in zip(same_meta, same_sims):
        # All diff-vehicle similarities for (vid, cam_i) against any camera
        neg_sims = [
            float(np.dot(features[vid][cam_i], features[other_vid][other_cam]))
            for other_vid in features
            if other_vid != vid
            for other_cam in features[other_vid]
        ]
        if neg_sims:
            rank1_total += 1
            if sim_pos > max(neg_sims):
                rank1_correct += 1

    if rank1_total:
        rank1 = 100.0 * rank1_correct / rank1_total
        print(f"\nRank-1 accuracy (same-vehicle beats all diff-vehicle): "
              f"{rank1_correct}/{rank1_total} = {rank1:.1f}%")

    # ------------------------------------------------------------------
    # Worst same-vehicle pairs (most confusing for the associator)
    # ------------------------------------------------------------------
    print("\nWorst 10 same-vehicle cross-camera pairs (lowest similarity):")
    sorted_same = sorted(zip(same_sims, same_meta))
    for sim, (vid, ci, cj) in sorted_same[:10]:
        print(f"  vehicle={vid:5d}  {ci} ↔ {cj}  sim={sim:.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(
        min(same_arr.min(), diff_arr.min()) - 0.05,
        max(same_arr.max(), diff_arr.max()) + 0.05,
        60,
    )

    ax.hist(same_arr, bins=bins, alpha=0.6, color="steelblue", label="Same vehicle, cross-camera", density=True)
    ax.hist(diff_arr, bins=bins, alpha=0.6, color="tomato",    label="Different vehicle",         density=True)

    ax.axvline(same_arr.mean(), color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Same mean = {same_arr.mean():.3f}")
    ax.axvline(diff_arr.mean(), color="tomato",    linestyle="--", linewidth=1.5,
               label=f"Diff mean = {diff_arr.mean():.3f}")

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"ReID feature similarity distributions — {args.seq}\n"
                 f"Separation = {separation:.4f}  |  Rank-1 = {rank1:.1f}%"
                 if rank1_total else f"ReID feature similarity distributions — {args.seq}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = out_dir / "reid_similarity_distributions.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")


if __name__ == "__main__":
    main()
