
import csv

import numpy as np
from tqdm import tqdm

from src.object_detection import BoundingBox, CarDetector, TemporalCarDetector
from src.models.adaptive_models import AdaptiveGrayGaussianModel
from src.video_source import VideoPartSource
from src.mask_postprocessing import Closing, Dilate, MaskPostprocess, Opening, RemoveSmallBlobs
from src.evaluation import evaluate_detections

import xml.etree.ElementTree as ET


def load_annotations(path: str):
    tree = ET.parse(path)
    boxes = tree.findall("track/box")
    boxes_per_frame = dict()
    for box in boxes:
        frame    = int(box.get("frame"))
        xtl      = float(box.get("xtl"))
        ytl      = float(box.get("ytl"))
        xbr      = float(box.get("xbr"))
        ybr      = float(box.get("ybr"))

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


CONFIGS = [
    {"n_frames": 1,  "threshold": 0.5},
    {"n_frames": 1,  "threshold": 0.3},
    {"n_frames": 1,  "threshold": 0.7},
    # {"n_frames": 5,  "threshold": 0.3},
    # {"n_frames": 5,  "threshold": 0.5},
    # {"n_frames": 5,  "threshold": 0.7},
    # {"n_frames": 10, "threshold": 0.3},
    # {"n_frames": 10, "threshold": 0.5},
    # {"n_frames": 10, "threshold": 0.7},
]


if __name__ == '__main__':
    ANNOTATIONS_PATH = "../ai_challenge_s03_c010-full_annotation.xml"
    VIDEO_PATH = "../AICity_data/AICity_data/train/S03/c010/vdo.avi"
    OUTPUT_CSV = "temporal_sweep_results.csv"

    train_source = VideoPartSource(VIDEO_PATH, 0.0, 0.25)
    test_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)

    model = AdaptiveGrayGaussianModel(3.0, mean_rho=0.05, variance_rho=0.05)
    annotations = load_annotations(ANNOTATIONS_PATH)

    print("Fitting model...")
    model.fit_from_source(train_source)

    postprocess = MaskPostprocess(
        Opening((5, 5)),
        Closing((5, 5)),
        Dilate((7, 7)),
        RemoveSmallBlobs(300),
    )

    # Precompute all postprocessed masks so we only run the model once
    print("Precomputing masks...")
    masks_and_ids = []
    for mask, (frame_id, _) in tqdm(
        zip(model.predict_from_source(test_source),
            enumerate(iter(test_source), start=test_source.start_frame)),
        total=test_source.n_frames,
    ):
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        masks_and_ids.append((frame_id, postprocess(mask)))

    gt_for_eval = {fid: boxes for fid, boxes in annotations.items()
                   if any(fid == mid for mid, _ in masks_and_ids)}

    base_detector = CarDetector(
        area=[1000, 100000],
        aspect_ratio=[0.1, 10.0],
        fill_ratio=[0.2, 1.0],
    )

    results = []
    for cfg in CONFIGS:
        detector = TemporalCarDetector(base_detector, **cfg)
        predictions = {}
        for frame_id, mask in masks_and_ids:
            predictions[frame_id] = detector.detect(mask)

        metrics = evaluate_detections(gt_for_eval, predictions)
        ap50 = metrics["AP50"]
        results.append({**cfg, "AP50": ap50})
        print(f"n_frames={cfg['n_frames']:>2}, threshold={cfg['threshold']:.1f}  =>  AP50={ap50:.4f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_frames", "threshold", "AP50"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")
