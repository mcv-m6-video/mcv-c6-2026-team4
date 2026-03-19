from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.multi_camera_associator import GlobalTrack, _normalize
from src.world_and_camera_tracking import WorldTrack

if TYPE_CHECKING:
    from src.camera_graph import CameraGraph


class ClusteringAssociator:
    """
    Order-independent multi-camera associator based on agglomerative clustering.

    Unlike the greedy CombinedAssociator, this builds the full N×N pairwise
    cost matrix before making any merge decisions, making it insensitive to
    processing order.

    Same-camera pairs are allowed for fragmentation repair (tracklets broken by
    brief occlusions or detector dropouts). They use a simpler proximity + time
    gap cost with a hard gate on maximum gap duration.

    Cost function
    -------------
    cost(i, j) = w_reid * reid_cost(i, j) + w_geo * geo_cost(i, j)

        reid_cost  = 1 − cosine_similarity(f_i, f_j)   ∈ [0, 1]
                     0.5 if either feature is missing (neutral)

        geo_cost, different cameras:
            if temporal overlap ≥ min_overlap_secs:
                mean world-distance at sampled timestamps / geo_scale
                (co-temporal: directly checks if two projections are of the same object)
            elif tracks are sequential (one ends before the other starts):
                clamp(endpoint_distance / (v_max * Δt), 0, 1)
                (speed-consistency: 0 = easily reachable, 1 = at the physical limit)
            else (small overlap < min_overlap_secs):
                clamp(min_endpoint_distance / geo_scale, 0, 1)

        geo_cost, same camera (fragmentation repair):
            clamp(endpoint_distance / geo_scale, 0, 1)
            hard gates:
                temporal overlap   → inf (two fragments cannot overlap in time)
                Δt > same_camera_max_gap → inf (gap too long; likely different occurrences)

    Hard inf gates (cost = inf, never merged):
        - Same-camera pairs with temporal overlap
        - Same-camera pairs with time gap > same_camera_max_gap
        - Cross-camera pairs blocked by camera_graph

    Parameters
    ----------
    distance_threshold:
        Clusters are merged only when their linkage distance is below this value.
        Equivalent role to match_threshold in CombinedAssociator.
    linkage:
        "average" (UPGMA) — recommended default; balanced, not too aggressive.
        "complete"        — stricter, use if you see over-merging.
        "single"          — too aggressive for this task, avoid.
    w_reid:
        Weight for the ReID appearance cost component.
    w_geo:
        Weight for the geometric cost component.
    geo_scale:
        Normaliser for spatial distances (world units). Used only in the
        sequential fallback branch (no temporal overlap). Set to the expected
        maximum meaningful inter-position distance between endpoints of two
        tracklets that could be the same car.
    n_sigma:
        Mahalanobis distance at which the co-temporal geo cost reaches 1.0
        (i.e. the cost ceiling for the co-temporal branch).  A value of 3
        means "three combined standard deviations apart = fully penalised".
        Lower values penalise spatial disagreement more aggressively.
        Because Mahalanobis distance is normalised by per-point uncertainty
        propagated through each camera's homography Jacobian, this parameter
        is sequence-agnostic — no per-sequence calibration required.
    v_max:
        Maximum vehicle speed in world units per second. Used for the
        speed-consistency cost on sequential cross-camera pairs.
    same_camera_max_gap:
        Hard gate (seconds). Same-camera pairs with a temporal gap above this
        are blocked. Set to a short value (e.g. 3–5 s) to avoid merging
        separate passes of the same camera.
    min_overlap_secs:
        Minimum temporal overlap (seconds) to use the co-temporal distance
        instead of the speed-consistency cost.
    n_overlap_samples:
        Number of evenly spaced timestamps sampled during the overlap interval
        when computing the co-temporal distance.
    camera_graph:
        Optional. If provided, sequential cross-camera pairs whose transition
        is declared infeasible get cost = inf.
    """

    def __init__(
        self,
        *,
        distance_threshold: float = 0.4,
        linkage: str = "average",
        w_reid: float = 1.0,
        w_geo: float = 0.0,
        geo_scale: float = 1.0,
        n_sigma: float = 3.0,
        v_max: float = 30.0,
        same_camera_max_gap: float = 5.0,
        min_overlap_secs: float = 1.0,
        n_overlap_samples: int = 10,
        camera_graph: CameraGraph | None = None,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.w_reid = w_reid
        self.w_geo = w_geo
        self.geo_scale = geo_scale
        self.n_sigma = n_sigma
        self.v_max = v_max
        self.same_camera_max_gap = same_camera_max_gap
        self.min_overlap_secs = min_overlap_secs
        self.n_overlap_samples = n_overlap_samples
        self.camera_graph = camera_graph

    # ------------------------------------------------------------------
    # Public API

    def associate(
        self,
        world_tracks: list[WorldTrack],
        plot_cost: bool = False,
        plot_path: str | None = None,
    ) -> list[GlobalTrack]:
        n = len(world_tracks)
        if n == 0:
            return []

        features = [
            _normalize(wt.reid_feature) if wt.reid_feature is not None else None
            for wt in world_tracks
        ]

        # Build symmetric N×N pairwise cost matrix.
        cost = np.full((n, n), np.inf)
        np.fill_diagonal(cost, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                c = self._pairwise_cost(
                    world_tracks[i], world_tracks[j], features[i], features[j]
                )
                cost[i, j] = cost[j, i] = c

        # sklearn AgglomerativeClustering with metric="precomputed" does not
        # tolerate inf values. Replace them with a sentinel that is larger than
        # any valid finite cost, so blocked pairs are never the closest pair.
        sentinel = self.w_reid + self.w_geo + 1.0
        finite_cost = np.where(np.isinf(cost), sentinel, cost)

        # Temporary diagnostic — cross-camera finite costs
        xc = cost[(~np.isinf(cost)) & (cost > 0)]
        if len(xc):
            print(f"  Cross-cam cost dist: min={xc.min():.3f}  "
                f"p25={np.percentile(xc,25):.3f}  median={np.median(xc):.3f}  "
                f"p75={np.percentile(xc,75):.3f}  max={xc.max():.3f}")

        if n == 1:
            labels = np.array([0])
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="precomputed",
                linkage=self.linkage,
            )
            labels = model.fit_predict(finite_cost)

        if plot_cost:
            self._plot_cost_matrix(cost, labels, plot_path)

        clusters: dict[int, list[WorldTrack]] = defaultdict(list)
        for wt, label in zip(world_tracks, labels):
            clusters[label].append(wt)

        return [
            GlobalTrack(global_id=gid, tracklets=tracklets)
            for gid, (_, tracklets) in enumerate(clusters.items(), start=1)
        ]

    # ------------------------------------------------------------------
    # Pairwise cost dispatcher

    def _pairwise_cost(
        self,
        wt_i: WorldTrack,
        wt_j: WorldTrack,
        fi: np.ndarray | None,
        fj: np.ndarray | None,
    ) -> float:
        reid = self._reid_cost(fi, fj)
        same_cam = wt_i.camera_id == wt_j.camera_id

        ts_i = self._time_span(wt_i)
        ts_j = self._time_span(wt_j)

        if ts_i is None or ts_j is None:
            # No timestamps available.
            # Same-camera without geo is ambiguous — block to be safe.
            if same_cam:
                return np.inf
            # Cross-camera: appearance only if w_geo == 0, else no signal → block.
            if self.w_geo == 0.0:
                return self.w_reid * reid
            return np.inf

        t_i0, t_i1 = ts_i
        t_j0, t_j1 = ts_j

        if same_cam:
            geo = self._same_camera_geo(wt_i, wt_j, t_i0, t_i1, t_j0, t_j1)
        else:
            geo = self._cross_camera_geo(wt_i, wt_j, t_i0, t_i1, t_j0, t_j1)

        if np.isinf(geo):
            return np.inf

        return self.w_reid * reid + self.w_geo * geo

    # ------------------------------------------------------------------
    # Geometric cost components

    def _same_camera_geo(
        self,
        wt_i: WorldTrack,
        wt_j: WorldTrack,
        t_i0: float, t_i1: float,
        t_j0: float, t_j1: float,
    ) -> float:
        """Fragmentation-repair cost for same-camera pairs."""
        # True temporal overlap → same car cannot be in two tracklets simultaneously.
        if t_i0 < t_j1 and t_j0 < t_i1:
            return np.inf

        # Determine which tracklet ends first and get the directed gap.
        if t_i1 <= t_j0:
            dt = t_j0 - t_i1
            d = self._endpoint_dist(wt_i, wt_j)   # i_end → j_start
        else:
            dt = t_i0 - t_j1
            d = self._endpoint_dist(wt_j, wt_i)   # j_end → i_start

        if dt > self.same_camera_max_gap:
            return np.inf

        return min(d / max(self.geo_scale, 1e-9), 1.0)

    def _cross_camera_geo(
        self,
        wt_i: WorldTrack,
        wt_j: WorldTrack,
        t_i0: float, t_i1: float,
        t_j0: float, t_j1: float,
    ) -> float:
        """Geometric cost for cross-camera pairs."""
        overlap_start = max(t_i0, t_j0)
        overlap_end   = min(t_i1, t_j1)
        overlap_dur   = overlap_end - overlap_start

        if overlap_dur >= self.min_overlap_secs:
            # Co-temporal regime: two cameras observe the car simultaneously.
            # Mean Mahalanobis distance at sampled timestamps measures whether
            # both tracklets project to the same physical location relative to
            # the combined homography uncertainty of each camera pair.
            # A value of 1 = within the noise floor; n_sigma = cost ceiling.
            d = self._co_temporal_dist(wt_i, wt_j, overlap_start, overlap_end)
            return min(d / max(self.n_sigma, 1e-9), 1.0)

        # Sequential / near-simultaneous handoff: determine direction.
        if t_i1 <= t_j0:
            dt = t_j0 - t_i1
            d  = self._endpoint_dist(wt_i, wt_j)
            from_cam, to_cam = wt_i.camera_id, wt_j.camera_id
        elif t_j1 <= t_i0:
            dt = t_i0 - t_j1
            d  = self._endpoint_dist(wt_j, wt_i)
            from_cam, to_cam = wt_j.camera_id, wt_i.camera_id
        else:
            # Overlap exists but shorter than min_overlap_secs — near-simultaneous
            # handoff. Use the more favourable directed endpoint distance.
            d1 = self._endpoint_dist(wt_i, wt_j)
            d2 = self._endpoint_dist(wt_j, wt_i)
            d, dt = min(d1, d2), 0.0
            from_cam, to_cam = wt_i.camera_id, wt_j.camera_id

        # Camera-graph feasibility hard gate (only for truly sequential pairs).
        if self.camera_graph is not None and dt > 0:
            if not self.camera_graph.is_transition_feasible(from_cam, to_cam, dt):
                return np.inf

        # Speed-consistency cost: fraction of maximum reachable distance used.
        # 0 = car is well within its speed budget, 1 = at the physical limit.
        if dt > 0 and self.v_max > 0:
            return min(d / (self.v_max * dt), 1.0)

        # No time gap (dt == 0) or v_max not set: fall back to scaled distance.
        return min(d / max(self.geo_scale, 1e-9), 1.0)

    # ------------------------------------------------------------------
    # Visualisation

    @staticmethod
    def _plot_cost_matrix(
        cost: np.ndarray,
        labels: np.ndarray,
        save_path: str | None = None,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Reorder rows/cols so tracks in the same cluster are contiguous
        # → clusters appear as blocks on the diagonal.
        order = np.argsort(labels, kind="stable")
        sorted_cost = cost[np.ix_(order, order)]

        display = sorted_cost.astype(float)

        fig_px = min(1600, max(600, len(labels) * 4))
        fig_in = fig_px / 100
        fig, ax = plt.subplots(figsize=(fig_in, fig_in))

        # Clip inf to 1.0 — all finite costs are in [0, 1] by construction,
        # so blocked pairs simply render as worst-cost red with no visual noise.
        display = np.clip(display, 0.0, 1.0)

        im = ax.imshow(display, cmap="RdBu_r", vmin=0, vmax=1,
                       aspect="equal", interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cost  (blue = compatible, red = incompatible / blocked)", fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Pairwise cost matrix — sorted by cluster", fontsize=9)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"  Cost matrix saved → {save_path}")
        else:
            fig.savefig("cost_matrix.png", dpi=100, bbox_inches="tight")
            print("  Cost matrix saved → cost_matrix.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Helpers

    def _co_temporal_dist(
        self,
        wt_i: WorldTrack,
        wt_j: WorldTrack,
        t_start: float,
        t_end: float,
    ) -> float:
        """
        Mean Mahalanobis world-space distance between the two tracklets at
        n_overlap_samples evenly spaced timestamps within [t_start, t_end].

        Uses WorldTrack.position_at() for linear interpolation and
        WorldPoint.mahalanobis_dist() for the distance, which normalises by
        the combined per-point uncertainty propagated from each camera's
        homography Jacobian.  A value of ~1 means "within the noise floor of
        the homography"; values well above n_sigma indicate clearly different
        physical locations.

        Returns inf if no valid position pairs can be obtained (e.g.
        single-observation tracks that do not support interpolation).
        """
        times = np.linspace(t_start, t_end, self.n_overlap_samples)
        dists = []
        for t in times:
            pi = wt_i.position_at(float(t))
            pj = wt_j.position_at(float(t))
            if pi is not None and pj is not None:
                dists.append(pi.mahalanobis_dist(pj))
        return float(np.mean(dists)) if dists else np.inf

    @staticmethod
    def _endpoint_dist(wt_from: WorldTrack, wt_to: WorldTrack) -> float:
        """World-space distance from the last observation of wt_from to the
        first observation of wt_to."""
        if not wt_from.world_observations or not wt_to.world_observations:
            return np.inf
        p0 = wt_from.world_observations[-1].world_point.as_array()
        p1 = wt_to.world_observations[0].world_point.as_array()
        return float(np.linalg.norm(p1 - p0))

    @staticmethod
    def _reid_cost(fi: np.ndarray | None, fj: np.ndarray | None) -> float:
        """1 − cosine_similarity. Returns 0.5 (neutral) if either feature is None."""
        if fi is None or fj is None:
            return 0.5
        sim = float(np.dot(fi, fj))
        return 1.0 - float(np.clip(sim, -1.0, 1.0))

    @staticmethod
    def _time_span(wt: WorldTrack) -> tuple[float, float] | None:
        """Returns (t_start, t_end) for the tracklet, or None if unavailable."""
        obs = wt.world_observations
        if not obs:
            return None
        t0 = obs[0].timestamp
        t1 = obs[-1].timestamp
        if t0 is None or t1 is None:
            return None
        return float(t0), float(t1)
