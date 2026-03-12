"""
Corridor density and geometry models.

Extracted from the original annotate_months.py and simulate_isabella_bird_mortality.py.
All functions are site-agnostic: corridors, blobs, and weights come from config.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .config import BlobDef, CorridorDef, SiteConfig

Point = Tuple[float, float]


# ── Vector helpers ────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def vec_add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def vec_sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def vec_mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def vec_len(a: Point) -> float:
    return math.hypot(a[0], a[1])


def vec_norm(a: Point) -> Point:
    length = vec_len(a)
    if length == 0:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def vec_perp(a: Point) -> Point:
    return (-a[1], a[0])


def rotate(v: Point, ang_rad: float) -> Point:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


# ── Bezier curves ─────────────────────────────────────────────────

def bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1 - t
    return (
        (u ** 3) * p0[0] + 3 * (u ** 2) * t * p1[0] + 3 * u * (t ** 2) * p2[0] + (t ** 3) * p3[0],
        (u ** 3) * p0[1] + 3 * (u ** 2) * t * p1[1] + 3 * u * (t ** 2) * p2[1] + (t ** 3) * p3[1],
    )


def bezier_deriv(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1 - t
    return (
        3 * (u ** 2) * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * (t ** 2) * (p3[0] - p2[0]),
        3 * (u ** 2) * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * (t ** 2) * (p3[1] - p2[1]),
    )


def build_curved_corridor(
    p0: Point, p3: Point, curv: float, seasonal_shift: float
) -> Tuple[Point, Point, Point, Point]:
    """
    Build a cubic Bezier from p0→p3 with curvature applied perpendicular
    to the corridor direction, plus a seasonal shift for month variation.
    """
    d = vec_sub(p3, p0)
    dn = vec_norm(d)
    n = vec_perp(dn)

    mid = vec_add(p0, vec_mul(d, 0.5))
    ctrl_offset = curv + seasonal_shift
    ctrl = vec_add(mid, vec_mul(n, ctrl_offset))

    p1 = vec_add(p0, vec_mul(vec_sub(ctrl, p0), 0.6))
    p2 = vec_add(p3, vec_mul(vec_sub(ctrl, p3), 0.6))
    return p0, p1, p2, p3


# ── 2D Gaussian ───────────────────────────────────────────────────

def gaussian2d(
    x: np.ndarray, y: np.ndarray,
    mx: float, my: float, sx: float, sy: float,
) -> np.ndarray:
    return np.exp(-(((x - mx) ** 2) / (2 * sx ** 2) + ((y - my) ** 2) / (2 * sy ** 2)))


# ── Corridor density field (for mortality simulation) ─────────────

def corridor_density(
    xy: np.ndarray,
    month_idx: int,
    cfg: SiteConfig,
) -> np.ndarray:
    """
    Compute bird density at each turbine position for a given month.

    Uses all corridors from config (rotated Gaussian cross-sections with
    curvature) plus density blobs, weighted by season.

    Args:
        xy: (N, 2) array of turbine positions in [0,1] normalized space
        month_idx: 0-based month index (0=Jan, 11=Dec)
        cfg: site configuration
    """
    x = xy[:, 0]
    y = xy[:, 1]

    t = month_idx / 11.0
    drift_x = 0.02 * math.sin(2 * math.pi * t)
    drift_y = 0.02 * math.cos(2 * math.pi * t)

    total = np.zeros_like(x)

    for corr in cfg.corridors:
        theta = math.radians(corr.angle_deg)

        # Rotate coordinates into corridor-local frame
        xr = (x - 0.5) * math.cos(theta) - (y - 0.5) * math.sin(theta)
        yr = (x - 0.5) * math.sin(theta) + (y - 0.5) * math.cos(theta)

        # Apply curvature
        bend = corr.curvature * np.sin(2 * math.pi * (xr + 0.5) * 1.2)
        yr_curved = yr - bend

        if corr.center_x is not None:
            x_center = corr.center_x + drift_x
            ns_curve = 0.03 * np.sin(2 * math.pi * (y + drift_y) * 1.1)
            profile = np.exp(-((x - (x_center + ns_curve)) ** 2) / (2 * (corr.sigma ** 2)))
        else:
            profile = np.exp(-(yr_curved ** 2) / (2 * (corr.sigma ** 2)))

        # Season-dependent weighting
        cal = cfg.monthly_calendar[month_idx]
        if cal.season == "spring":
            w = corr.weight_spring
        elif cal.season == "fall":
            w = corr.weight_fall
        else:
            w = corr.weight_default

        total += w * profile

    # Density blobs (roosting, water features, etc.)
    blob_sum = np.zeros_like(x)
    for blob in cfg.density_blobs:
        bx, by = blob.center
        sx, sy = blob.spread
        blob_sum += blob.weight * gaussian2d(x, y, bx + drift_x, by + drift_y, sx, sy)

    if cfg.density_blobs:
        total += 0.35 * blob_sum

    return total


def dist_point_to_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Distance from point P to line segment A→B."""
    AB = B - A
    denom = np.dot(AB, AB) + 1e-12
    t = np.dot(P - A, AB) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = A + t * AB
    return float(np.linalg.norm(P - proj))
