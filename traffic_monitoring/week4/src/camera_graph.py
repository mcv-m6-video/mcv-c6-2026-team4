from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.dataset import Camera, Sequence


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class CameraEdge:
    """
    A directed edge from camera `src` to camera `dst`.

    Parameters
    ----------
    allowed:
        Hard gate.  False → the transition is structurally impossible
        (e.g. cameras on different road networks with no connecting path).
    distance:
        World-space distance between the two cameras' ROI centroids.
        float('nan') when centroids could not be computed.
    t_min, t_max:
        Feasible travel-time window (seconds).  Derived automatically from
        distance and assumed vehicle speed bounds, or set manually.
        A time gap outside [t_min, t_max] makes the transition implausible.

    Placeholders for future extension
    ----------------------------------
    exit_zone_id, entry_zone_id:
        Identifiers of the specific road-exit / road-entry zones that this
        edge represents.  Currently unused (None); reserved for when you want
        to distinguish "car left c001 via the south road" from "car left c001
        via the north road" and route it accordingly.
    """
    allowed:       bool
    distance:      float
    t_min:         float
    t_max:         float
    exit_zone_id:  str | None = None   # future: road-exit zone on src camera
    entry_zone_id: str | None = None   # future: road-entry zone on dst camera


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class CameraGraph:
    """
    Directed graph of camera-to-camera transition feasibility.

    Each edge encodes whether a car can realistically travel from the
    observable region of camera `src` to that of camera `dst`, and within
    what time window that journey is plausible given vehicle speed bounds.

    Usage with CombinedAssociator
    ------------------------------
    Pass the graph as ``camera_graph=graph`` to ``CombinedAssociator``.
    The associator will gate candidate matches through
    ``is_transition_feasible`` before computing the matching cost, replacing
    the coarser ``same_camera_filter`` flag.

    Construction
    ------------
    The easiest entry point is the class method ``from_sequence`` (or
    ``from_cameras``), which auto-computes ROI centroids in world space and
    derives time windows from speed bounds.  Manual overrides let you correct
    topology that geometry alone cannot infer (one-way streets, physical
    barriers, etc.).

    Example
    -------
    >>> graph = CameraGraph.from_sequence(
    ...     seq,
    ...     v_min=0.00001,        # world units / second — tune to your scale
    ...     v_max=0.0002,
    ...     max_distance=0.003,   # cameras further apart → not connected
    ...     overrides={
    ...         ("c001", "c004"): {"allowed": False},          # one-way road
    ...         ("c003", "c005"): {"t_min": 5.0, "t_max": 60.0},
    ...     },
    ... )
    >>> graph.is_transition_feasible("c001", "c003", time_gap=12.5)
    True
    """

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: dict[tuple[str, str], CameraEdge] = {}

    # ------------------------------------------------------------------
    # Mutation

    def add_camera(self, camera_id: str) -> None:
        self._nodes.add(camera_id)

    def set_edge(self, src: str, dst: str, edge: CameraEdge) -> None:
        self._nodes.add(src)
        self._nodes.add(dst)
        self._edges[(src, dst)] = edge

    # ------------------------------------------------------------------
    # Query

    def get_edge(self, src: str, dst: str) -> CameraEdge | None:
        return self._edges.get((src, dst))

    def cameras(self) -> set[str]:
        return set(self._nodes)

    def is_transition_feasible(
        self,
        src: str,
        dst: str,
        time_gap: float | None = None,
    ) -> bool:
        """
        Returns True if a car transitioning from `src` to `dst` with the
        given `time_gap` (seconds) is considered feasible.

        Rules
        -----
        - If either camera is not in the graph: True (no opinion, don't gate).
        - If the directed edge src→dst does not exist: False (unknown = forbidden).
        - If the edge is marked ``allowed=False``: False.
        - If ``time_gap`` is None (timestamps unavailable): True when allowed
          (topology is fine, just can't check the time window).
        - Otherwise: True iff ``t_min <= time_gap <= t_max``.
        """
        if src not in self._nodes or dst not in self._nodes:
            return True   # camera not in graph → no gating

        edge = self._edges.get((src, dst))
        if edge is None:
            return False  # no known path → forbidden

        if not edge.allowed:
            return False

        if time_gap is not None:
            return edge.t_min <= time_gap <= edge.t_max

        return True  # allowed but can't check time window

    def summary(self) -> str:
        """Human-readable summary of the graph for debugging."""
        lines = [f"CameraGraph ({len(self._nodes)} cameras, {len(self._edges)} directed edges)"]
        for (src, dst), e in sorted(self._edges.items()):
            status = "OK" if e.allowed else "BLOCKED"
            lines.append(
                f"  {src} → {dst}  [{status}]  "
                f"dist={e.distance:.4g}  "
                f"t=[{e.t_min:.1f}, {e.t_max:.1f}]s"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Construction from dataset

    @classmethod
    def from_sequence(
        cls,
        seq: Sequence,
        v_min: float,
        v_max: float,
        max_distance: float | None = None,
        allow_same_camera: bool = False,
        overrides: dict[tuple[str, str], dict] | None = None,
        pixel_step: int = 10,
        iqr_factor: float = 1.5,
    ) -> CameraGraph:
        """Build a graph from a ``Sequence`` object (convenience wrapper)."""
        return cls.from_cameras(
            list(seq.cameras.values()),
            v_min=v_min,
            v_max=v_max,
            max_distance=max_distance,
            allow_same_camera=allow_same_camera,
            overrides=overrides,
            pixel_step=pixel_step,
            iqr_factor=iqr_factor,
        )

    @classmethod
    def from_cameras(
        cls,
        cameras: list[Camera],
        v_min: float,
        v_max: float,
        max_distance: float | None = None,
        allow_same_camera: bool = False,
        overrides: dict[tuple[str, str], dict] | None = None,
        pixel_step: int = 10,
        iqr_factor: float = 1.5,
    ) -> CameraGraph:
        """
        Build a directed camera graph from a list of ``Camera`` objects.

        For each ordered pair (src, dst) of distinct cameras:
          - The world-space centroid of each camera's ROI is computed via the
            inverse homography (same pixel→world convention used throughout
            the pipeline).
          - ``distance`` = Euclidean distance between centroids.
          - If ``max_distance`` is set and ``distance > max_distance``:
            the edge is marked ``allowed=False``.
          - Otherwise: ``t_min = distance / v_max``,
                        ``t_max = distance / v_min``.

        Parameters
        ----------
        cameras:
            List of ``Camera`` objects (from ``AICityDataset``).
        v_min, v_max:
            Minimum / maximum expected vehicle speed in **world units per
            second**.  Must match the scale of your homography's world
            coordinates.  Tip: run ``CameraGraph.from_sequence`` with a test
            value and inspect ``graph.summary()`` — the ``t_min / t_max``
            values should be in the range of seconds, not milliseconds or hours.
        max_distance:
            If the centroid-to-centroid distance exceeds this, the edge is
            marked as not allowed.  None = no distance limit (rely on time
            gating only).
        allow_same_camera:
            If True, self-edges (c001→c001) are added with t_min=0, t_max=∞.
            Useful when a single camera's tracklets can be re-associated
            (e.g. car re-enters FOV after leaving).
        overrides:
            Manual corrections applied after auto-computation.
            Keys are ``(src_id, dst_id)`` tuples; values are dicts with any
            subset of ``CameraEdge`` field names to override.
            Example: ``{("c001", "c004"): {"allowed": False}}``
        pixel_step:
            Subsampling step for ROI pixels when computing centroids.
            Higher = faster but less accurate.
        iqr_factor:
            IQR multiplier for outlier rejection of projected ROI pixels.
            1.5 is the standard "mild outlier" threshold.
        """
        graph = cls()

        # -- Step 1: compute world-space centroid for each camera --
        centroids: dict[str, np.ndarray] = {}
        for cam in cameras:
            graph.add_camera(cam.id)
            c = _roi_world_centroid(cam, pixel_step, iqr_factor)
            if c is not None:
                centroids[cam.id] = c
            else:
                print(f"[CameraGraph] WARN: could not compute ROI centroid for {cam.id}")

        # -- Step 2: build directed edges for every ordered pair --
        cam_ids = [cam.id for cam in cameras]
        for src in cam_ids:
            for dst in cam_ids:
                if src == dst:
                    if allow_same_camera:
                        graph.set_edge(src, dst, CameraEdge(
                            allowed=True, distance=0.0,
                            t_min=0.0, t_max=float("inf"),
                        ))
                    continue

                # Both centroids must be known to compute distance
                if src not in centroids or dst not in centroids:
                    graph.set_edge(src, dst, CameraEdge(
                        allowed=True, distance=float("nan"),
                        t_min=0.0, t_max=float("inf"),
                    ))
                    continue

                dist = float(np.linalg.norm(centroids[src] - centroids[dst]))

                if max_distance is not None and dist > max_distance:
                    graph.set_edge(src, dst, CameraEdge(
                        allowed=False, distance=dist,
                        t_min=float("inf"), t_max=float("inf"),
                    ))
                    continue

                t_min = dist / v_max if v_max > 0.0 else 0.0
                t_max = dist / v_min if v_min > 0.0 else float("inf")
                graph.set_edge(src, dst, CameraEdge(
                    allowed=True, distance=dist, t_min=t_min, t_max=t_max,
                ))

        # -- Step 3: apply manual overrides --
        if overrides:
            for (src, dst), kwargs in overrides.items():
                existing = graph.get_edge(src, dst)
                if existing is not None:
                    merged = {**dataclasses.asdict(existing), **kwargs}
                else:
                    merged = kwargs
                graph.set_edge(src, dst, CameraEdge(**merged))

        return graph


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _roi_world_centroid(
    cam: Camera,
    pixel_step: int,
    iqr_factor: float,
) -> np.ndarray | None:
    """
    Returns the (x, y) world-space centroid of a camera's ROI.

    Uses the inverse homography for pixel→world projection (same convention
    as world_and_camera_tracking.py — calibration.txt stores H as world→pixel).
    """
    roi = cv2.imread(str(cam.roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return None

    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    sub = binary[::pixel_step, ::pixel_step]
    rows, cols = np.where(sub > 0)
    if len(rows) == 0:
        return None

    img_pts = np.column_stack([cols * pixel_step, rows * pixel_step]).astype(np.float64)

    H_inv = np.linalg.inv(cam.calibration.homography)
    ones = np.ones((len(img_pts), 1))
    pts_h = np.hstack([img_pts, ones])
    world_h = (H_inv @ pts_h.T).T
    w = world_h[:, 2]
    valid = np.abs(w) > 1e-9
    if not np.any(valid):
        return None

    world_pts = world_h[valid, :2] / w[valid, np.newaxis]

    # IQR outlier rejection
    mask = np.ones(len(world_pts), dtype=bool)
    for axis in range(2):
        vals = world_pts[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            mask &= (vals >= q1 - iqr_factor * iqr) & (vals <= q3 + iqr_factor * iqr)
    world_pts = world_pts[mask]

    return world_pts.mean(axis=0) if len(world_pts) > 0 else None
