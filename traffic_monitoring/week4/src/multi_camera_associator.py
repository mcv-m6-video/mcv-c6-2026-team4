from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.world_and_camera_tracking import WorldObservation, WorldPoint, WorldTrack

if TYPE_CHECKING:
    from src.camera_graph import CameraGraph


# ---------------------------------------------------------------------------
# GlobalTrack
# ---------------------------------------------------------------------------

@dataclass
class GlobalTrack:
    """
    A vehicle trajectory assembled from one or more single-camera tracklets.

    `tracklets` is ordered by the time of association (generally chronological).
    """
    global_id: int
    tracklets: list[WorldTrack] = field(default_factory=list)

    def cameras(self) -> list[str]:
        """Camera IDs that contributed a tracklet, in association order."""
        return [wt.camera_id for wt in self.tracklets]

    def fused_position_at(self, t: float) -> WorldPoint | None:
        """
        Inverse-variance-weighted average of all tracklets' interpolated
        positions at global timestamp `t`.

        Tracklets that cannot provide a position at `t` are ignored.
        Returns None if no tracklet covers `t`.
        """
        positions: list[np.ndarray] = []
        weights: list[float] = []

        for wt in self.tracklets:
            wp = wt.position_at(t)
            if wp is not None and wp.sigma_sq > 0.0:
                positions.append(wp.as_array())
                weights.append(1.0 / wp.sigma_sq)

        if not positions:
            return None

        w = np.asarray(weights)
        w /= w.sum()
        pos = np.average(np.stack(positions), axis=0, weights=w)
        fused_sigma_sq = 1.0 / sum(weights)
        return WorldPoint(x=float(pos[0]), y=float(pos[1]), sigma_sq=fused_sigma_sq)

    def all_world_observations(self) -> list[WorldObservation]:
        """Flat list of every world observation across all tracklets, sorted by timestamp."""
        obs = [wo for wt in self.tracklets for wo in wt.world_observations]
        obs.sort(key=lambda o: (o.timestamp is None, o.timestamp or 0, o.frame_idx))
        return obs


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class MultiCameraAssociator(Protocol):
    """Interface for offline multi-camera track association."""

    def associate(self, world_tracks: list[WorldTrack]) -> list[GlobalTrack]:
        """
        Groups `world_tracks` (one per local camera tracklet) into
        `GlobalTrack` objects, each representing a single physical vehicle.
        """
        ...


# ---------------------------------------------------------------------------
# Internal gallery state (private to CombinedAssociator)
# ---------------------------------------------------------------------------

@dataclass
class _GalleryEntry:
    global_track: GlobalTrack
    feature: np.ndarray | None       # representative ReID embedding (EMA-updated)
    last_world_point: WorldPoint | None
    last_timestamp: float | None
    last_velocity: np.ndarray | None  # world units / second at track end
    camera_ids: set[str]
    last_camera_id: str = ""         # camera that contributed the most recent tracklet
    original_camera_id: str = ""     # camera that created this entry (never changes)


# ---------------------------------------------------------------------------
# CombinedAssociator
# ---------------------------------------------------------------------------

class CombinedAssociator:
    """
    Greedy offline multi-camera associator with a configurable combined cost.

    Tracklets are processed in chronological order (by start timestamp).
    Each tracklet is matched to the best existing global track in the gallery,
    or seeded as a new global track if no eligible match is found.

    Combined cost
    -------------
    cost = w_appearance * appearance_cost
         + w_spatial    * (spatial_dist   / spatial_scale)
         + w_temporal   * (temporal_gap   / temporal_scale)
         + w_velocity   * (velocity_error / spatial_scale)

    Cost components (all non-negative):
        appearance_cost  : 1 − cosine_similarity(f_new, f_gallery)  ∈ [0, 1]
        spatial_dist     : Euclidean distance between end of gallery track
                           and start of incoming tracklet (world units)
        temporal_gap     : |t_end_gallery − t_start_new|  (seconds)
        velocity_error   : distance between constant-velocity-predicted
                           position of gallery track and actual first
                           position of new tracklet (world units)

    Setting a weight to 0 disables that component entirely (its cost is not
    computed). To replicate the appearance-only behaviour of the original
    mtmc.py, keep all weights at their defaults.

    Parameters
    ----------
    w_appearance, w_spatial, w_temporal, w_velocity:
        Weights for each cost component.
    match_threshold:
        Combined cost below which a match is accepted.
    same_camera_filter:
        If True, a tracklet will never be merged into a global track that
        already contains a tracklet from the same camera.
    feature_momentum:
        EMA weight applied to the *old* gallery feature on a successful match.
        0.0 = always replace with new feature.
        1.0 = never update (keep original).
        Default 0.8 matches the original mtmc.py.
        Only meaningful when w_appearance > 0.
    spatial_scale:
        Divides raw spatial distance to normalise it.
        Set to the expected maximum inter-camera distance in world units.
    temporal_scale:
        Divides raw time gap to normalise it.
        Set to the expected maximum transition time between cameras (seconds).
    max_time_gap:
        Hard gate: candidate pairs whose temporal gap exceeds this (seconds)
        are rejected before the cost is even computed.
    max_spatial_gap:
        Hard gate: candidate pairs whose spatial distance exceeds this
        (world units) are rejected before the cost is computed.
    """

    def __init__(
        self,
        *,
        w_appearance: float = 1.0,
        w_spatial: float = 0.0,
        w_temporal: float = 0.0,
        w_velocity: float = 0.0,
        match_threshold: float = 0.4,
        same_camera_filter: bool = True,
        camera_graph: CameraGraph | None = None,
        feature_momentum: float = 0.8,
        spatial_scale: float = 1.0,
        temporal_scale: float = 1.0,
        max_time_gap: float = float("inf"),
        max_spatial_gap: float = float("inf"),
        debug_costs: bool = False,
        sequential_cameras: list[str] | None = None,
    ) -> None:
        self.w_appearance = w_appearance
        self.w_spatial = w_spatial
        self.w_temporal = w_temporal
        self.w_velocity = w_velocity
        self.match_threshold = match_threshold
        self.same_camera_filter = same_camera_filter
        self.camera_graph = camera_graph
        self.feature_momentum = feature_momentum
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.max_time_gap = max_time_gap
        self.max_spatial_gap = max_spatial_gap
        self.debug_costs = debug_costs
        self.sequential_cameras = sequential_cameras

    # ------------------------------------------------------------------
    # Public API

    def associate(self, world_tracks: list[WorldTrack]) -> list[GlobalTrack]:
        """
        Associates `world_tracks` into global tracks.

        When `sequential_cameras` is set, tracklets are processed camera by
        camera in the given order (all of cam[0] first, then cam[1], …),
        matching mtmc.py's sequential strategy.  Otherwise, all tracklets are
        sorted by start timestamp (default, better for spatial/temporal costs).
        """
        gallery: list[_GalleryEntry] = []
        next_global_id = 1

        ordered = (
            self._sort_by_camera(world_tracks, self.sequential_cameras)
            if self.sequential_cameras is not None
            else self._sort_by_start(world_tracks)
        )
        for wt in ordered:
            best_entry, best_cost = self._find_best_match(wt, gallery)

            if best_entry is not None:
                best_entry.global_track.tracklets.append(wt)
                self._update_entry(best_entry, wt)
            else:
                gt = GlobalTrack(global_id=next_global_id, tracklets=[wt])
                gallery.append(self._make_entry(gt, wt))
                next_global_id += 1

        return [e.global_track for e in gallery]

    # ------------------------------------------------------------------
    # Matching

    def _find_best_match(
        self, wt: WorldTrack, gallery: list[_GalleryEntry]
    ) -> tuple[_GalleryEntry | None, float]:
        best_entry: _GalleryEntry | None = None
        best_cost = float("inf")

        for entry in gallery:
            if self.camera_graph is not None:
                # Graph-based topology gate: replaces same_camera_filter when a
                # graph is present.  Uses the most recent camera in the gallery
                # track as the transition source.
                t_gap = self._time_gap(wt, entry)
                if not self.camera_graph.is_transition_feasible(
                    entry.last_camera_id, wt.camera_id, t_gap
                ):
                    continue
            elif self.same_camera_filter:
                # Sequential mode replicates mtmc.py: block only if the
                # incoming tracklet is from the *original* creator camera
                # (cam_id is never updated in mtmc.py, so multiple tracklets
                # from the same non-creator camera can merge into one global).
                # Default mode: block if camera already contributed at all.
                if self.sequential_cameras is not None:
                    if wt.camera_id == entry.original_camera_id:
                        continue
                elif wt.camera_id in entry.camera_ids:
                    continue
            cost = self._compute_cost(wt, entry)
            if cost < best_cost:
                best_cost, best_entry = cost, entry

        if best_cost < self.match_threshold:
            return best_entry, best_cost
        return None, best_cost

    def _compute_cost(self, wt: WorldTrack, entry: _GalleryEntry) -> float:
        # Hard gates: reject cheaply before computing full cost
        if self.max_time_gap < float("inf") or self.w_temporal > 0:
            t_gap = self._time_gap(wt, entry)
            if t_gap is not None and t_gap > self.max_time_gap:
                return float("inf")

        if self.max_spatial_gap < float("inf") or self.w_spatial > 0:
            s_dist = self._spatial_dist(wt, entry)
            if s_dist is not None and s_dist > self.max_spatial_gap:
                return float("inf")

        cost = 0.0
        dbg: dict = {}  # populated only when debug_costs=True

        app_raw = self._appearance_cost(wt, entry) if self.w_appearance > 0 else None
        if app_raw is not None:
            app_w = self.w_appearance * app_raw
            cost += app_w
            if self.debug_costs:
                dbg["appearance"] = {"raw": app_raw, "weighted": app_w}

        s_dist = self._spatial_dist(wt, entry) if self.w_spatial > 0 else None
        if s_dist is not None:
            s_norm = s_dist / self.spatial_scale
            s_w = self.w_spatial * s_norm
            cost += s_w
            if self.debug_costs:
                dbg["spatial"] = {"raw": s_dist, "norm": s_norm, "weighted": s_w}

        t_gap = self._time_gap(wt, entry) if self.w_temporal > 0 else None
        if t_gap is not None:
            t_norm = t_gap / self.temporal_scale
            t_w = self.w_temporal * t_norm
            cost += t_w
            if self.debug_costs:
                dbg["temporal"] = {"raw": t_gap, "norm": t_norm, "weighted": t_w}

        v_err = self._velocity_error(wt, entry) if self.w_velocity > 0 else None
        if v_err is not None:
            v_norm = v_err / self.spatial_scale
            v_w = self.w_velocity * v_norm
            cost += v_w
            if self.debug_costs:
                dbg["velocity"] = {"raw": v_err, "norm": v_norm, "weighted": v_w}

        if self.debug_costs:
            src = f"{entry.last_camera_id}/g{entry.global_track.global_id}"
            dst = f"{wt.camera_id}/t{wt.track_id}"
            parts = []
            for name, vals in dbg.items():
                if "norm" in vals:
                    parts.append(f"{name}: {vals['raw']:.4g} → {vals['norm']:.4g} → {vals['weighted']:.4g}")
                else:
                    parts.append(f"{name}: {vals['raw']:.4g} → {vals['weighted']:.4g}")
            print(f"  [{src} → {dst}]  total={cost:.4f}  |  " + "  |  ".join(parts))

        return cost

    # ------------------------------------------------------------------
    # Cost components

    def _appearance_cost(self, wt: WorldTrack, entry: _GalleryEntry) -> float:
        """
        1 − cosine_similarity(f_new, f_gallery).

        If either feature is absent:
          - appearance is the sole active criterion → return inf (cannot match)
          - other criteria are also active          → return 0.0 (no signal, neutral)
        """
        f_new = wt.reid_feature
        f_gal = entry.feature
        if f_new is None or f_gal is None:
            only_appearance = (
                self.w_spatial == 0 and self.w_temporal == 0 and self.w_velocity == 0
            )
            return float("inf") if only_appearance else 0.0
        sim = float(np.dot(f_new, f_gal))
        return 1.0 - float(np.clip(sim, -1.0, 1.0))

    def _spatial_dist(self, wt: WorldTrack, entry: _GalleryEntry) -> float | None:
        """Euclidean distance: end of gallery track → start of incoming tracklet."""
        if not wt.world_observations or entry.last_world_point is None:
            return None
        start = wt.world_observations[0].world_point.as_array()
        end = entry.last_world_point.as_array()
        return float(np.linalg.norm(start - end))

    def _time_gap(self, wt: WorldTrack, entry: _GalleryEntry) -> float | None:
        """Absolute time gap between end of gallery track and start of incoming tracklet."""
        t_start = wt.world_observations[0].timestamp if wt.world_observations else None
        if t_start is None or entry.last_timestamp is None:
            return None
        return abs(t_start - entry.last_timestamp)

    def _velocity_error(self, wt: WorldTrack, entry: _GalleryEntry) -> float | None:
        """
        Euclidean distance between the constant-velocity prediction of the
        gallery track (extrapolated over the time gap) and the actual first
        position of the incoming tracklet.
        """
        if entry.last_world_point is None or entry.last_velocity is None:
            return None
        t_gap = self._time_gap(wt, entry)
        if t_gap is None or not wt.world_observations:
            return None
        predicted = entry.last_world_point.as_array() + entry.last_velocity * t_gap
        actual = wt.world_observations[0].world_point.as_array()
        return float(np.linalg.norm(predicted - actual))

    # ------------------------------------------------------------------
    # Gallery management

    def _make_entry(self, gt: GlobalTrack, wt: WorldTrack) -> _GalleryEntry:
        feature = _normalize(wt.reid_feature) if wt.reid_feature is not None else None
        last_wp = wt.world_observations[-1].world_point if wt.world_observations else None
        last_ts = wt.world_observations[-1].timestamp if wt.world_observations else None
        last_vel = wt.velocity_at(-1)
        return _GalleryEntry(
            global_track=gt,
            feature=feature,
            last_world_point=last_wp,
            last_timestamp=last_ts,
            last_velocity=last_vel,
            camera_ids={wt.camera_id},
            last_camera_id=wt.camera_id,
            original_camera_id=wt.camera_id,
        )

    def _update_entry(self, entry: _GalleryEntry, wt: WorldTrack) -> None:
        """Update gallery state after a successful match."""
        # Feature: exponential moving average
        new_feat = _normalize(wt.reid_feature) if wt.reid_feature is not None else None
        if new_feat is not None:
            if entry.feature is None:
                entry.feature = new_feat
            else:
                blended = self.feature_momentum * entry.feature + (1.0 - self.feature_momentum) * new_feat
                entry.feature = _normalize(blended)

        # Spatial/temporal state: always advance to the latest observation
        if wt.world_observations:
            entry.last_world_point = wt.world_observations[-1].world_point
            entry.last_timestamp = wt.world_observations[-1].timestamp
            entry.last_velocity = wt.velocity_at(-1)

        entry.camera_ids.add(wt.camera_id)
        entry.last_camera_id = wt.camera_id

    # ------------------------------------------------------------------
    # Utilities

    @staticmethod
    def _sort_by_start(world_tracks: list[WorldTrack]) -> list[WorldTrack]:
        def key(wt: WorldTrack) -> float:
            if wt.world_observations:
                ts = wt.world_observations[0].timestamp
                if ts is not None:
                    return ts
                return float(wt.world_observations[0].frame_idx)
            return 0.0
        return sorted(world_tracks, key=key)

    @staticmethod
    def _sort_by_camera(
        world_tracks: list[WorldTrack], camera_order: list[str]
    ) -> list[WorldTrack]:
        """
        Sort tracklets camera-by-camera in the given order, then by start
        timestamp within each camera.  Tracklets from cameras not in
        `camera_order` are appended at the end, sorted by timestamp.
        Replicates mtmc.py's sequential per-camera processing strategy.
        """
        rank = {cam: i for i, cam in enumerate(camera_order)}

        def key(wt: WorldTrack):
            cam_rank = rank.get(wt.camera_id, len(camera_order))
            ts = 0.0
            if wt.world_observations:
                t = wt.world_observations[0].timestamp
                ts = t if t is not None else float(wt.world_observations[0].frame_idx)
            return (cam_rank, ts)

        return sorted(world_tracks, key=key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0.0 else v
