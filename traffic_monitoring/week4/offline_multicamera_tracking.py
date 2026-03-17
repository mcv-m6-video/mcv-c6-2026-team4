import itertools
from typing import Any, Callable, Iterable

import cv2

from src.bounding_box import BoundingBox
from src import dataset
from src.single_camera_tracker import SingleCameraTracker, Track
from src.synced_video_source import SyncedCameraSource
from src.world_and_camera_tracking import ContactPoint, ContactPointFn, WorldTrack, project_tracks
from src.multi_camera_associator import CombinedAssociator, GlobalTrack
from src.reid_feature_extractor import FeatureExtractor, compute_features

CAR_CLASS = 0


class OfflineMulticameraTracker:
    """
    Full offline multi-camera tracking pipeline:
        1. Per-camera detection + single-camera tracking
        2. Ground-plane projection (pixel → world via homography)
        3. Optional ReID / appearance feature extraction
        4. Multi-camera association → GlobalTracks

    Parameters
    ----------
    tracker_factory:
        Callable that returns a *fresh* SingleCameraTracker for each camera.
        A new instance is created per camera so internal state never leaks
        across cameras.
    detector:
        Any callable that accepts a frame (np.ndarray) and returns a YOLO-
        style result with a `.boxes` attribute.
    associator:
        A `CombinedAssociator` (or any `MultiCameraAssociator`) instance.
    contact_strategy:
        How to pick the ground-contact pixel from a bounding box.
    feature_extractor:
        Optional extractor called on each WorldTrack to populate
        `wt.reid_feature`.  If None, association uses only geometry.
    confidence:
        Minimum YOLO detection confidence.
    """

    def __init__(
        self,
        tracker_factory: Callable[[], SingleCameraTracker],
        detector,
        associator: CombinedAssociator,
        contact_strategy: ContactPoint | ContactPointFn = ContactPoint.BOTTOM_CENTER,
        feature_extractor: FeatureExtractor | None = None,
        confidence: float = 0.45,
    ):
        self.tracker_factory = tracker_factory
        self.detector = detector
        self.associator = associator
        self.contact_strategy = contact_strategy
        self.feature_extractor = feature_extractor
        self.confidence = confidence

    # ------------------------------------------------------------------

    def track(self, cameras: list[dataset.Camera]) -> list[GlobalTrack]:
        tracks_per_camera = self._track_per_camera(cameras)
        projected = self._project_tracks(cameras, tracks_per_camera)

        all_world_tracks: list[WorldTrack] = list(
            itertools.chain.from_iterable(projected.values())
        )

        if self.feature_extractor is not None:
            compute_features(all_world_tracks, self.feature_extractor)

        return self.associator.associate(all_world_tracks)

    # ------------------------------------------------------------------
    # Internal steps

    def _track_per_camera(
        self, cameras: list[dataset.Camera]
    ) -> dict[str, list[Track]]:
        tracks_per_camera: dict[str, list[Track]] = {}
        for camera in cameras:
            roi_mask = self._load_roi_mask(camera.roi_path)
            # Fresh tracker per camera — prevents state leaking across cameras
            tracker = self.tracker_factory()
            video_source = SyncedCameraSource(camera)

            for idx, frame, timestamp in video_source:
                results = self.detector(frame, verbose=False, conf=self.confidence)[0]
                detections = [
                    BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=float(box.conf[0]))
                    for box in self._car_filter(self._roi_filter(results.boxes, roi_mask))
                    for x1, y1, x2, y2 in [box.xyxy[0].cpu().numpy()]
                ]
                tracker.update(detections, frame, idx, timestamp)

            tracks_per_camera[camera.id] = tracker.finalize()
            print(f"  [{camera.id}] {len(tracks_per_camera[camera.id])} local tracks")

        return tracks_per_camera

    def _project_tracks(
        self,
        cameras: list[dataset.Camera],
        tracks_per_camera: dict[str, list[Track]],
    ) -> dict[str, list[WorldTrack]]:
        return {
            camera.id: project_tracks(
                tracks_per_camera[camera.id],
                camera.id,
                camera.calibration.homography,
                camera.calibration.reprojection_error,
                self.contact_strategy,
            )
            for camera in cameras
        }

    # ------------------------------------------------------------------
    # Detection helpers

    def _load_roi_mask(self, path):
        roi_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if roi_mask is not None:
            _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            print(f"Warning: ROI at {path} not found.")
        return roi_mask

    def _car_filter(self, boxes: Iterable[Any]) -> Iterable[Any]:
        return (box for box in boxes if int(box.cls[0]) == CAR_CLASS)

    def _roi_filter(self, boxes: Iterable[Any], roi_mask) -> Iterable[Any]:
        if roi_mask is None:
            yield from boxes
            return
        height, width = roi_mask.shape
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            corners = [
                (ix1, iy1), (ix1, iy2), (ix2, iy1), (ix2, iy2),
            ]
            if any(
                0 <= cx < width and 0 <= cy < height and roi_mask[cy, cx] < 255
                for cx, cy in corners
            ):
                continue
            yield box
