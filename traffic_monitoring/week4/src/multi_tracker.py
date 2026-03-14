from scipy.optimize import linear_sum_assignment
import numpy as np

def _iou(b1, b2):
    """Calculates Intersection over Union (IoU) between two BoundingBoxes."""
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

class MultiTracker:
    """
    A unified tracker interface allowing you to switch between 
    'max_overlap' (greedy matching) and 'sort' (Hungarian matching + optional Kalman).
    """
    def __init__(self, method='max_overlap', max_age=5, iou_threshold=0.45):
        assert method in ['max_overlap', 'sort'], "Method must be 'max_overlap' or 'sort'"
        self.method = method
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        
        self.next_id = 1
        self.active_tracks = {}
        self.lost_tracks = {}
        self.finished_tracks = []

    def update(self, detections, frame_img, frame_idx):
        if self.method == 'max_overlap':
            self._update_max_overlap(detections, frame_img, frame_idx)
        elif self.method == 'sort':
            self._update_sort(detections, frame_img, frame_idx)

    def _extract_crop(self, img, bbox):
        h, w = img.shape[:2]
        x1, y1 = max(0, int(bbox.left)), max(0, int(bbox.top))
        x2, y2 = min(w, int(bbox.right)), min(h, int(bbox.bottom))
        return img[y1:y2, x1:x2].copy()

    def _update_max_overlap(self, detections, frame_img, frame_idx):
        """Your original Greedy IoU logic."""
        unmatched_dets = list(detections)
        new_active_tracks = {}

        # 1. Match active tracks (Greedy)
        for track_id, info in self.active_tracks.items():
            best_iou, best_idx = 0, -1
            for i, det in enumerate(unmatched_dets):
                iou = _iou(info["current_bbox"], det)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            if best_iou > self.iou_threshold:
                matched_det = unmatched_dets.pop(best_idx)
                crop = self._extract_crop(frame_img, matched_det)
                info["history"].append({"frame": frame_idx, "bbox": matched_det, "img": crop})
                info["current_bbox"] = matched_det
                info["last_seen"] = frame_idx
                new_active_tracks[track_id] = info
            else:
                self.lost_tracks[track_id] = info

        # 2. Match lost tracks (Greedy)
        still_unmatched = []
        for det in unmatched_dets:
            best_iou, best_lost_id = 0, -1
            for lost_id, info in list(self.lost_tracks.items()):
                if frame_idx - info["last_seen"] > self.max_age:
                    self.finished_tracks.append({"history": info["history"]})
                    del self.lost_tracks[lost_id]
                    continue
                
                iou = _iou(info["current_bbox"], det)
                if iou > best_iou:
                    best_iou, best_lost_id = iou, lost_id

            if best_iou > self.iou_threshold:
                crop = self._extract_crop(frame_img, det)
                info = self.lost_tracks.pop(best_lost_id)
                info["history"].append({"frame": frame_idx, "bbox": det, "img": crop})
                info["current_bbox"] = det
                info["last_seen"] = frame_idx
                new_active_tracks[best_lost_id] = info
            else:
                still_unmatched.append(det)

        # 3. Initialize new tracks
        self._init_new_tracks(still_unmatched, frame_img, frame_idx, new_active_tracks)
        self.active_tracks = new_active_tracks

    def _update_sort(self, detections, frame_img, frame_idx):
        """SORT-style logic using the Hungarian Algorithm for optimal matching."""
        # Combine active and eligible lost tracks for matching
        track_ids = []
        track_bboxes = []
        
        for tid, info in self.active_tracks.items():
            track_ids.append((tid, 'active'))
            track_bboxes.append(info["current_bbox"])
            
        for tid, info in list(self.lost_tracks.items()):
            if frame_idx - info["last_seen"] > self.max_age:
                self.finished_tracks.append({"history": info["history"]})
                del self.lost_tracks[tid]
            else:
                track_ids.append((tid, 'lost'))
                track_bboxes.append(info["current_bbox"])

        new_active_tracks = {}
        unmatched_dets = list(detections)

        if len(track_bboxes) > 0 and len(detections) > 0:
            # Build IoU Cost Matrix
            iou_matrix = np.zeros((len(track_bboxes), len(detections)), dtype=np.float32)
            for t, trk_box in enumerate(track_bboxes):
                for d, det_box in enumerate(detections):
                    iou_matrix[t, d] = _iou(trk_box, det_box)

            # Hungarian Algorithm minimizes cost, so we invert IoU (1 - IoU)
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            unmatched_det_indices = set(range(len(detections)))

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > self.iou_threshold:
                    matched_det = detections[c]
                    unmatched_det_indices.discard(c)
                    
                    tid, status = track_ids[r]
                    crop = self._extract_crop(frame_img, matched_det)
                    
                    # Retrieve the correct track info
                    info = self.active_tracks.pop(tid) if status == 'active' else self.lost_tracks.pop(tid)
                    
                    info["history"].append({"frame": frame_idx, "bbox": matched_det, "img": crop})
                    info["current_bbox"] = matched_det
                    info["last_seen"] = frame_idx
                    new_active_tracks[tid] = info

            # Rebuild unmatched detections list based on what the Hungarian algorithm rejected
            unmatched_dets = [detections[i] for i in unmatched_det_indices]

        # Any active tracks that weren't matched go to lost
        for tid, info in self.active_tracks.items():
            self.lost_tracks[tid] = info
            
        # Initialize new tracks for unmatched detections
        self._init_new_tracks(unmatched_dets, frame_img, frame_idx, new_active_tracks)
        self.active_tracks = new_active_tracks

    def _init_new_tracks(self, unmatched_dets, frame_img, frame_idx, new_active_tracks):
        for det in unmatched_dets:
            crop = self._extract_crop(frame_img, det)
            new_active_tracks[self.next_id] = {
                "current_bbox": det, 
                "last_seen": frame_idx, 
                "history": [{"frame": frame_idx, "bbox": det, "img": crop}]
            }
            self.next_id += 1

    def finalize(self):
        for info in self.active_tracks.values():
            self.finished_tracks.append({"history": info["history"]})
        for info in self.lost_tracks.values():
            self.finished_tracks.append({"history": info["history"]})
        self.active_tracks.clear()
        self.lost_tracks.clear()
        return self.finished_tracks