"""
Turbine layout generation and avoidance models.

Extracted from simulate_isabella_bird_mortality.py and annotate_months.py.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .config import SiteConfig
from .corridors import Point, vec_add, vec_len, vec_mul, vec_norm, rotate


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_turbine_layout(cfg: SiteConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pseudo-spatial turbine layout from config clusters.

    If the config has lat/lon turbine positions (from CSV import), those
    are projected into [0,1] space instead of using cluster generation.

    Returns:
        turbine_ids: (N,) int array of IDs 1..N
        xy: (N, 2) array of positions in [0, 1] normalized space
    """
    if cfg.turbine_latlon is not None:
        from .geo import bounding_box, latlon_to_normalized
        lats = cfg.turbine_latlon[:, 0]
        lons = cfg.turbine_latlon[:, 1]
        bbox = bounding_box(lats, lons)
        xy = latlon_to_normalized(lats, lons, bbox)
        n = len(xy)
        turbine_ids = np.arange(1, n + 1, dtype=int)
        return turbine_ids, xy

    n = cfg.turbine_count
    rng = np.random.default_rng(cfg.layout_seed)

    parts = []
    allocated = 0

    for cluster in cfg.clusters:
        count = int(n * cluster.fraction)
        allocated += count
        pts = rng.normal(
            loc=cluster.center,
            scale=cluster.spread,
            size=(count, 2),
        )
        parts.append(pts)

    # Remaining turbines placed uniformly
    remaining = n - allocated
    if remaining > 0:
        bg = rng.uniform(low=(0.10, 0.15), high=(0.90, 0.85), size=(remaining, 2))
        parts.append(bg)

    xy = np.vstack(parts)[:n]
    xy[:, 0] = np.clip(xy[:, 0], 0.0, 1.0)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, 1.0)

    turbine_ids = np.arange(1, n + 1, dtype=int)
    return turbine_ids, xy


def turbine_avoidance_factor(xy: np.ndarray) -> np.ndarray:
    """
    K-NN clustering-based avoidance factor.

    Turbines in dense clusters get higher avoidance (birds learn to steer
    away from areas with many rotors). Uses the 6 nearest neighbors.

    Returns per-turbine avoidance values in [0, ~0.65].
    """
    n = len(xy)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.30])

    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)

    k = min(6, n - 1)
    knn = np.sort(d, axis=1)[:, :k]
    mean_knn = np.mean(knn, axis=1)

    z = (np.median(mean_knn) - mean_knn) / (np.std(mean_knn) + 1e-9)
    avoid = 0.65 * sigmoid(1.4 * z)
    return avoid


def turbine_deflect(
    pos: Point,
    direction: Point,
    turbines: List[Tuple[int, int]],
    intensity: float,
    rng: np.random.Generator,
) -> Tuple[Point, Point]:
    """
    Deflect a bird's position and heading away from nearby turbines.

    Used in the visual corridor renderer to show arrows bending around
    turbine locations. Strength scales with migration intensity.
    """
    R = 160.0
    push_max = 24.0 * (0.4 + 0.6 * intensity)
    turn_max = math.radians(14.0) * (0.4 + 0.6 * intensity)

    best = None
    best_dist = 1e9
    for tx, ty in turbines:
        dvec = (pos[0] - tx, pos[1] - ty)
        dist = math.hypot(dvec[0], dvec[1])
        if dist < best_dist:
            best_dist = dist
            best = (tx, ty, dvec, dist)

    if best is None:
        return pos, direction

    tx, ty, dvec, dist = best
    if dist > R:
        return pos, direction

    away = vec_norm(dvec)
    strength = 1.0 - dist / R
    push = push_max * strength
    pos2 = vec_add(pos, vec_mul(away, push))

    dirn = vec_norm(direction)
    cross = dirn[0] * away[1] - dirn[1] * away[0]
    sign = 1.0 if cross > 0 else -1.0
    jitter = rng.normal(0.0, math.radians(2.0))
    ang = sign * turn_max * strength + jitter
    dir2 = rotate(dirn, ang)
    return pos2, dir2
