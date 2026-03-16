from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class Calibration:
    """Camera calibration data parsed from calibration.txt."""
    homography: np.ndarray          # (3, 3) image-to-world homography matrix
    reprojection_error: float
    intrinsics: Optional[np.ndarray] = None   # (3, 3) camera intrinsic matrix
    distortion: Optional[np.ndarray] = None   # (1, 4) distortion coefficients


@dataclass
class Camera:
    """All data associated with a single camera."""
    id: str                          # e.g. "c001"
    sequence_id: str                 # e.g. "S01"
    path: Path                       # directory root for this camera

    video_path: Path                 # vdo.avi
    roi_path: Path                   # roi.jpg  (binary ROI mask)
    calibration: Calibration

    start_timestamp: float           # seconds; from cam_timestamp/SXX.txt
    frame_count: int                 # from cam_framenum/SXX.txt

    gt_path: Optional[Path]          # gt/gt.txt (None if not present)
    det_paths: dict = field(default_factory=dict)   # {"yolo3": Path, "ssd512": Path, ...}
    mtsc_paths: dict = field(default_factory=dict)  # {"deepsort_yolo3": Path, ...}
    segm_path: Optional[Path] = None                # segm/segm_mask_rcnn.txt


@dataclass
class Sequence:
    """A scenario (e.g. S01) containing multiple cameras."""
    id: str                          # e.g. "S01"
    path: Path                       # train/S01/
    cameras: dict                    # {camera_id: Camera}
    location_map_path: Optional[Path] = None   # cam_loc/S01.png

    def __getitem__(self, cam_id: str) -> "Camera":
        return self.cameras[cam_id]


class AICityDataset:
    """
    Loads the AI City Challenge 2022 dataset from a root directory.

    Usage:
        dataset = AICityDataset("/path/to/AI_CITY_CHALLENGE_2022_TRAIN")
        for seq in dataset.sequences.values():
            for cam in seq.cameras.values():
                print(cam.video_path, cam.calibration.homography)

        # Flat access helpers
        cam = dataset["S01"]["c001"]
        all_cameras = dataset.cameras()
    """

    # cam_loc maps: some scenarios share a single map image
    _LOC_MAP_ALIASES = {
        "S03": "S0345",
        "S04": "S0345",
        "S05": "S0345",
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.sequences: dict[str, Sequence] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __getitem__(self, seq_id: str) -> Sequence:
        return self.sequences[seq_id]

    def cameras(self) -> list:
        """Flat list of every Camera across all sequences."""
        return [cam for seq in self.sequences.values() for cam in seq.cameras.values()]

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self):
        train_root = self.root / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"Expected a 'train/' folder under {self.root}")

        # Pre-load per-scenario timestamp and frame-count tables
        timestamps = self._load_scenario_table(self.root / "cam_timestamp")
        frame_counts = self._load_scenario_table(self.root / "cam_framenum")

        for seq_dir in sorted(train_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_id = seq_dir.name  # "S01", "S03", ...
            cameras = {}

            for cam_dir in sorted(seq_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                cam_id = cam_dir.name  # "c001", "c002", ...

                cam = self._load_camera(
                    cam_dir,
                    cam_id,
                    seq_id,
                    timestamps.get(seq_id, {}).get(cam_id, 0.0),
                    frame_counts.get(seq_id, {}).get(cam_id, -1),
                )
                cameras[cam_id] = cam

            loc_map = self._find_location_map(seq_id)
            self.sequences[seq_id] = Sequence(
                id=seq_id,
                path=seq_dir,
                cameras=cameras,
                location_map_path=loc_map,
            )

    def _load_camera(
        self,
        cam_dir: Path,
        cam_id: str,
        seq_id: str,
        start_timestamp: float,
        frame_count: int,
    ) -> "Camera":
        # Required files
        video_path = cam_dir / "vdo.avi"
        roi_path = cam_dir / "roi.jpg"
        calibration = self._parse_calibration(cam_dir / "calibration.txt")

        # Optional ground truth
        gt_path = cam_dir / "gt" / "gt.txt"
        gt_path = gt_path if gt_path.exists() else None

        # Detection files: det/det_<method>.txt  ->  key = <method>
        det_paths = {}
        det_dir = cam_dir / "det"
        if det_dir.exists():
            for p in sorted(det_dir.glob("det_*.txt")):
                key = p.stem[len("det_"):]  # "yolo3", "ssd512", "mask_rcnn"
                det_paths[key] = p

        # MTSC tracking files: mtsc/mtsc_<tracker>_<detector>.txt  ->  key = <tracker>_<detector>
        mtsc_paths = {}
        mtsc_dir = cam_dir / "mtsc"
        if mtsc_dir.exists():
            for p in sorted(mtsc_dir.glob("mtsc_*.txt")):
                key = p.stem[len("mtsc_"):]  # "deepsort_yolo3", "moana_mask_rcnn", ...
                mtsc_paths[key] = p

        # Segmentation file
        segm_path = cam_dir / "segm" / "segm_mask_rcnn.txt"
        segm_path = segm_path if segm_path.exists() else None

        return Camera(
            id=cam_id,
            sequence_id=seq_id,
            path=cam_dir,
            video_path=video_path,
            roi_path=roi_path,
            calibration=calibration,
            start_timestamp=start_timestamp,
            frame_count=frame_count,
            gt_path=gt_path,
            det_paths=det_paths,
            mtsc_paths=mtsc_paths,
            segm_path=segm_path,
        )

    # ------------------------------------------------------------------
    # File parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_scenario_table(directory: Path) -> dict:
        """
        Parses all SXX.txt files in a directory.
        Each file has lines: <camera_id> <value>
        Returns: {"S01": {"c001": value, ...}, ...}
        """
        table = {}
        if not directory.exists():
            return table
        for txt_file in sorted(directory.glob("*.txt")):
            seq_id = txt_file.stem  # "S01"
            entries = {}
            for line in txt_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    cam_id, raw_value = parts
                    try:
                        entries[cam_id] = float(raw_value)
                    except ValueError:
                        entries[cam_id] = raw_value
            table[seq_id] = entries
        return table

    @staticmethod
    def _parse_calibration(path: Path) -> Calibration:
        """
        Parses calibration.txt.

        Expected format:
            Homography matrix: v1 v2 v3;v4 v5 v6;v7 v8 v9
            Reprojection error: <float>
            [Intrinsic matrix: v1 v2 v3;v4 v5 v6;v7 v8 v9]
            [Distortion coefficients: v1 v2 v3 v4]
        """
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        homography = None
        reprojection_error = 0.0
        intrinsics = None
        distortion = None

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("Homography matrix:"):
                raw = line.split(":", 1)[1].strip()
                homography = AICityDataset._parse_matrix_3x3(raw)

            elif line.startswith("Reprojection error:"):
                reprojection_error = float(line.split(":", 1)[1].strip())

            elif line.startswith("Intrinsic matrix:") or line.startswith("Intrinsics:"):
                raw = line.split(":", 1)[1].strip()
                intrinsics = AICityDataset._parse_matrix_3x3(raw)

            elif line.startswith("Distortion"):
                raw = line.split(":", 1)[1].strip()
                distortion = np.array([float(v) for v in raw.split()], dtype=np.float64)

        if homography is None:
            raise ValueError(f"No homography matrix found in {path}")

        return Calibration(
            homography=homography,
            reprojection_error=reprojection_error,
            intrinsics=intrinsics,
            distortion=distortion,
        )

    @staticmethod
    def _parse_matrix_3x3(raw: str) -> np.ndarray:
        """Parses a 3x3 matrix from 'v1 v2 v3;v4 v5 v6;v7 v8 v9'."""
        rows = raw.split(";")
        return np.array(
            [[float(v) for v in row.split()] for row in rows],
            dtype=np.float64,
        )

    def _find_location_map(self, seq_id: str) -> Optional[Path]:
        loc_dir = self.root / "cam_loc"
        alias = self._LOC_MAP_ALIASES.get(seq_id, seq_id)
        for candidate in [seq_id, alias]:
            p = loc_dir / f"{candidate}.png"
            if p.exists():
                return p
        return None
