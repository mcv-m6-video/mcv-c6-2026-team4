from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from src.bounding_box import BoundingBox
from src.single_camera_tracker import Track, TrackObservation


# ---------------------------------------------------------------------------
# Contact-point strategy
# ---------------------------------------------------------------------------

class ContactPoint(Enum):
    """Built-in strategies for picking the ground-contact pixel of a bounding box."""
    BOTTOM_CENTER = auto()
    CENTER = auto()

# A custom strategy is any callable (BoundingBox) -> (u, v)
ContactPointFn = Callable[[BoundingBox], tuple[float, float]]


def _resolve_contact(strategy: ContactPoint | ContactPointFn) -> ContactPointFn:
    if callable(strategy) and not isinstance(strategy, ContactPoint):
        return strategy
    if strategy == ContactPoint.BOTTOM_CENTER:
        return lambda b: ((b.left + b.right) / 2.0, float(b.bottom))
    if strategy == ContactPoint.CENTER:
        return lambda b: ((b.left + b.right) / 2.0, (b.top + b.bottom) / 2.0)
    raise ValueError(f"Unknown contact-point strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Homography projection helpers
# ---------------------------------------------------------------------------

def _project_point(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Apply a 3×3 homography H to pixel (u, v), returning world (x, y)."""
    p = H @ np.array([u, v, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def _homography_jacobian(H: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    Analytic 2×2 Jacobian of the homography projection at pixel (u, v).

    Given the homogeneous projection:
        [p0, p1, p2]^T = H @ [u, v, 1]^T
        x_w = p0/p2,  y_w = p1/p2

    The partial derivatives are:
        dx_w/du = (H[0,0]*p2 - p0*H[2,0]) / p2²
        dx_w/dv = (H[0,1]*p2 - p0*H[2,1]) / p2²
        dy_w/du = (H[1,0]*p2 - p1*H[2,0]) / p2²
        dy_w/dv = (H[1,1]*p2 - p1*H[2,1]) / p2²
    """
    p = H @ np.array([u, v, 1.0], dtype=np.float64)
    p0, p1, p2 = p
    p2_sq = p2 ** 2
    return np.array([
        [(H[0, 0] * p2 - p0 * H[2, 0]) / p2_sq, (H[0, 1] * p2 - p0 * H[2, 1]) / p2_sq],
        [(H[1, 0] * p2 - p1 * H[2, 0]) / p2_sq, (H[1, 1] * p2 - p1 * H[2, 1]) / p2_sq],
    ], dtype=np.float64)


def _propagate_sigma_sq(H: np.ndarray, u: float, v: float, sigma_pixel_sq: float) -> float:
    """
    Propagates isotropic pixel uncertainty σ²_pixel through the homography Jacobian
    to produce a scalar (isotropic) world-space variance.

    Derivation:
        Σ_world = J @ (σ²_pixel · I) @ J^T = σ²_pixel · J @ J^T

    The Frobenius norm squared of J equals trace(J^T @ J), so:
        σ²_world = σ²_pixel · ‖J‖²_F / 2

    Dividing by 2 averages the variance across both world axes, keeping the
    result scalar while correctly reflecting the local scale of the projection.

    This is more informative than a fixed global σ because near vanishing
    points the Jacobian norm grows large, correctly expressing that a 1-pixel
    error in image space maps to a large world-space uncertainty.
    """
    J = _homography_jacobian(H, u, v)
    return sigma_pixel_sq * float(np.sum(J ** 2)) / 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorldPoint:
    """A 2-D world position with isotropic uncertainty."""
    x: float
    y: float
    sigma_sq: float   # isotropic world-space variance σ²

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def mahalanobis_dist(self, other: WorldPoint) -> float:
        """
        Scalar Mahalanobis distance between two world points under their
        combined (averaged) isotropic covariance.
        """
        sigma_sq = (self.sigma_sq + other.sigma_sq) / 2.0
        if sigma_sq <= 0.0:
            return float("inf")
        diff = self.as_array() - other.as_array()
        return float(np.sqrt(np.dot(diff, diff) / sigma_sq))


@dataclass
class WorldObservation:
    """One frame of a track projected into world coordinates."""
    camera_obs: TrackObservation   # reference to the original camera-space observation
    world_point: WorldPoint

    # Convenience pass-throughs so callers don't have to drill into camera_obs
    @property
    def frame_idx(self) -> int:
        return self.camera_obs.frame_idx

    @property
    def timestamp(self) -> float | None:
        return self.camera_obs.timestamp

    @property
    def bbox(self) -> BoundingBox:
        return self.camera_obs.bbox


@dataclass
class WorldTrack:
    """
    A single-camera track enriched with world-space projections.

    The original `Track` is kept intact and unmodified. `world_observations`
    is a parallel list aligned 1-to-1 with `track.history`, so indices are
    always consistent between camera space and world space.
    """
    camera_id: str
    track: Track
    world_observations: list[WorldObservation]
    reid_feature: np.ndarray | None = None   # optional per-tracklet ReID embedding (L2-normalised)

    # ------------------------------------------------------------------
    # Identity / convenience

    @property
    def track_id(self) -> int:
        return self.track.track_id

    def world_points(self) -> list[WorldPoint]:
        return [wo.world_point for wo in self.world_observations]

    def timestamps(self) -> list[float | None]:
        return [wo.timestamp for wo in self.world_observations]

    # ------------------------------------------------------------------
    # Kinematics

    def velocity_at(self, idx: int = -1) -> np.ndarray | None:
        """
        Estimates instantaneous velocity (world units / second) at observation
        index `idx` using central differences (one-sided at the boundaries).

        Returns None if the track has fewer than 2 observations or if
        timestamps are not available.
        """
        obs = self.world_observations
        n = len(obs)
        if n < 2:
            return None

        if idx < 0:
            idx = n + idx

        if idx == 0:
            a, b = obs[0], obs[1]
        elif idx == n - 1:
            a, b = obs[-2], obs[-1]
        else:
            a, b = obs[idx - 1], obs[idx + 1]

        t0, t1 = a.timestamp, b.timestamp
        if t0 is None or t1 is None or t1 == t0:
            return None

        dp = b.world_point.as_array() - a.world_point.as_array()
        return dp / (t1 - t0)

    def position_at(self, t: float) -> WorldPoint | None:
        """
        Linearly interpolates the world position (and σ²) at global timestamp `t`.

        Returns None when `t` is outside the track's time span or any
        timestamp is missing.
        """
        obs = self.world_observations
        if len(obs) < 2:
            return None

        ts = [o.timestamp for o in obs]
        if any(ti is None for ti in ts):
            return None

        if t < ts[0] or t > ts[-1]:
            return None

        for i in range(len(obs) - 1):
            t0, t1 = ts[i], ts[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                wp0 = obs[i].world_point
                wp1 = obs[i + 1].world_point
                return WorldPoint(
                    x=wp0.x + alpha * (wp1.x - wp0.x),
                    y=wp0.y + alpha * (wp1.y - wp0.y),
                    sigma_sq=wp0.sigma_sq + alpha * (wp1.sigma_sq - wp0.sigma_sq),
                )

        return None


# ---------------------------------------------------------------------------
# Projection API
# ---------------------------------------------------------------------------

def project_track(
    track: Track,
    camera_id: str,
    homography: np.ndarray,
    reprojection_error: float,
    contact: ContactPoint | ContactPointFn = ContactPoint.BOTTOM_CENTER,
) -> WorldTrack:
    """
    Projects every observation in `track` to world coordinates, producing a
    `WorldTrack` with a parallel list of `WorldObservation` objects.

    Parameters
    ----------
    track:
        A single-camera track (typically from `SingleCameraTracker.finalize()`).
    camera_id:
        The camera identifier, e.g. ``"c001"``.
    homography:
        3×3 image-to-world homography from ``Calibration.homography``.
    reprojection_error:
        Per-camera RMS calibration error in pixels (``Calibration.reprojection_error``).
        Used as σ_pixel; propagated through the Jacobian to yield per-point
        world-space uncertainty.
    contact:
        How to pick the ground-contact pixel from a bounding box.
        Pass a ``ContactPoint`` enum value or any callable
        ``(BoundingBox) -> (u: float, v: float)``.
    """
    contact_fn = _resolve_contact(contact)
    sigma_pixel_sq = reprojection_error ** 2
    world_obs: list[WorldObservation] = []

    # calibration.txt stores H as world→pixel; invert once for pixel→world projection.
    H_inv = np.linalg.inv(homography)

    for cam_obs in track.history:
        u, v = contact_fn(cam_obs.bbox)
        x, y = _project_point(H_inv, u, v)
        sigma_sq = _propagate_sigma_sq(H_inv, u, v, sigma_pixel_sq)
        world_obs.append(WorldObservation(
            camera_obs=cam_obs,
            world_point=WorldPoint(x=x, y=y, sigma_sq=sigma_sq),
        ))

    return WorldTrack(camera_id=camera_id, track=track, world_observations=world_obs)


def project_tracks(
    tracks: list[Track],
    camera_id: str,
    homography: np.ndarray,
    reprojection_error: float,
    contact: ContactPoint | ContactPointFn = ContactPoint.BOTTOM_CENTER,
) -> list[WorldTrack]:
    """Convenience wrapper: project a list of tracks from the same camera."""
    return [
        project_track(t, camera_id, homography, reprojection_error, contact)
        for t in tracks
    ]
