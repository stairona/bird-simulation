"""
Multi-factor collision probability model.

Extracted from bird_sim.py. Computes per-timestep strike probability
based on weather, night conditions, altitude overlap, and avoidance.
"""

from __future__ import annotations

import numpy as np

from .config import CollisionParams, SeasonDef


def per_step_collision_prob(
    season: SeasonDef,
    is_night: bool,
    inside_risk_zone: bool,
    params: CollisionParams,
) -> float:
    """
    Probability that a bird is struck in one simulation timestep.

    Factors:
        - base_strike_prob: baseline probability when bird is in rotor zone
        - weather_risk: seasonal weather multiplier
        - night_risk_mult: elevated risk during nocturnal flight
        - altitude_match_prob: probability bird is at rotor height
        - avoidance: fraction of birds that successfully dodge (higher = safer)
    """
    if not inside_risk_zone:
        return 0.0

    p = params.base_strike_prob
    p *= season.weather_risk
    if is_night:
        p *= params.night_risk_mult
    p *= params.altitude_match_prob
    p *= (1.0 - params.avoidance)
    return float(np.clip(p, 0.0, 0.25))
