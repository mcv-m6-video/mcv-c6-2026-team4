
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.object_detection import BoundingBox


def evaluate_detections(
    gt_per_frame: dict[int, list[BoundingBox]],
    pred_per_frame: dict[int, list[BoundingBox]],
) -> dict[str, float]:
    all_frame_ids = sorted(set(gt_per_frame) | set(pred_per_frame))

    gt_coco: dict = {
        "images": [{"id": fid} for fid in all_frame_ids],
        "annotations": [],
        "categories": [{"id": 1, "name": "car"}],
    }

    ann_id = 1
    for fid, boxes in gt_per_frame.items():
        for bbox in boxes:
            x = float(bbox.left)
            y = float(bbox.top)
            w = float(bbox.right - bbox.left)
            h = float(bbox.bottom - bbox.top)
            gt_coco["annotations"].append({
                "id": ann_id,
                "image_id": fid,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    pred_list = []
    for fid, boxes in pred_per_frame.items():
        for bbox in boxes:
            x = float(bbox.left)
            y = float(bbox.top)
            w = float(bbox.right - bbox.left)
            h = float(bbox.bottom - bbox.top)
            pred_list.append({
                "image_id": fid,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "score": 1.0,
            })

    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()

    if not pred_list:
        return {k: 0.0 for k in (
            "AP", "AP50", "AP75",
            "AP_small", "AP_medium", "AP_large",
            "AR_max1", "AR_max10", "AR_max100",
            "AR_small", "AR_medium", "AR_large",
        )}

    coco_dt = coco_gt.loadRes(pred_list)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP":        stats[0],
        "AP50":      stats[1],
        "AP75":      stats[2],
        "AP_small":  stats[3],
        "AP_medium": stats[4],
        "AP_large":  stats[5],
        "AR_max1":   stats[6],
        "AR_max10":  stats[7],
        "AR_max100": stats[8],
        "AR_small":  stats[9],
        "AR_medium": stats[10],
        "AR_large":  stats[11],
    }

def show_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name} = {value}")
