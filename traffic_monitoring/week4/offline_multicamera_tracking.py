
import itertools
from typing import Any, Iterable

import cv2

from src.bounding_box import BoundingBox
from src.multi_tracker import MultiTracker
from src import dataset
from src.single_camera_tracker import SingleCameraTracker, Track
from src.synced_video_source import SyncedCameraSource
from src.world_and_camera_tracking import ContactPoint, ContactPointFn, project_tracks
from src.multi_camera_associator import CombinedAssociator

CAR_CLASS = 0

class OfflineMulticameraTracker:
    def __init__(self, detector, single_camera_tracker: SingleCameraTracker, contact_strategy: ContactPoint | ContactPointFn, associator: CombinedAssociator):
        self.detector = detector
        self.single_camera_tracker = single_camera_tracker
        self.contact_strategy = contact_strategy
        self.associator = associator

    def _load_roi_mask(self, path):
        roi_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is not None:
            _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            print(f"Warning: ROI at {path} not found.")
        
        return roi_mask

    def _car_filter(self, raw_bboxes: Iterable[Any]):
        for box in raw_bboxes:
            if int(box.cls[0]) == CAR_CLASS:
                yield box

    def _roi_filter(self, raw_bboxes: Iterable[Any], roi_mask):
        for bbox in raw_bboxes:
            if roi_mask is None:
                yield bbox
            
            x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            height, width = roi_mask.shape
            
            is_outlier = False
                        
            if 0 <= ix1 < width:
                if 0 <= iy1 < height and roi_mask[iy1, ix1] < 255:
                    is_outlier = True
                if 0 <= iy2 < height and roi_mask[iy2, ix1] < 255:
                    is_outlier = True
            
            if 0 <= ix2 < width:
                if 0 <= iy1 < height and roi_mask[iy1, ix2] < 255:
                    is_outlier = True
                if 0 <= iy2 < height and roi_mask[iy2, ix2] < 255:
                    is_outlier = True
            
            if not is_outlier:
                yield bbox 


    def _track_per_camera(self, cameras: list[dataset.Camera]):
        tracks_per_camera = dict()
        for camera in cameras:
            roi_mask = self._load_roi_mask(camera.roi_path)
            video_source = SyncedCameraSource(camera)
            for idx, frame, timestamp in iter(video_source):
                detection_results = self.detector(frame)
                detections = []
                for bbox in self._roi_filter(self._car_filter(detection_results.boxes), roi_mask):
                    conf = float(bbox.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()
                    detections.append(BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=conf))
                self.single_camera_tracker.update(detections, frame, idx, timestamp)
                
            tracks = self.single_camera_tracker.finalize()
            tracks_per_camera[camera.id] = tracks
            
        return tracks_per_camera
    
    def _project_tracks(self, cameras: list[dataset.Camera], tracks_per_camera: dict[str, list[Track]]):
        projected_tracks_per_camera = dict()
        for camera in cameras:
            tracks = tracks_per_camera[camera.id]
            projected_tracks = project_tracks(tracks, camera.id, camera.calibration.homography, camera.calibration.reprojection_error, self.contact_strategy)
            projected_tracks_per_camera[camera.id] = projected_tracks
            
        return projected_tracks_per_camera
    
    def track(self, cameras: list[Any]) -> list[Any]:
        tracks_per_camera = self._track_per_camera(cameras)
        projected_tracks_per_camera = self._project_tracks(cameras, tracks_per_camera)
        all_world_tracks = itertools.chain.from_iterable(projected_tracks_per_camera.values())
        global_tracks = self.associator.associate(all_world_tracks)
        return global_tracks
        
        
        
    
