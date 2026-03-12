"""
Phase 2 — Agent-based bird collision simulation.

Generalized from bird_sim.py. Spawns individual birds (migrants following
corridors, residents doing random walks), steps them through the world,
and checks proximity to turbines for collision events.

All parameters (corridors, turbines, seasons, collision knobs) from config.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..core.config import SiteConfig
from ..core.corridors import dist_point_to_segment
from ..core.collision import per_step_collision_prob
from ..core.calendar import build_month_to_season


def _build_corridors_from_config(cfg: SiteConfig) -> List[Dict]:
    """
    Convert config corridors to the agent sim format:
    line segments with p0/p1 endpoints and sigma.
    """
    import math
    W, H = cfg.simulation.agent.world_size
    corridors = []

    for corr in cfg.corridors:
        theta = math.radians(corr.angle_deg)
        dx = math.cos(theta)
        dy = math.sin(theta)

        if corr.center is not None:
            cx, cy = corr.center
            # Scale to world coords
            cx *= W
            cy *= H
        elif corr.center_x is not None:
            cx = corr.center_x * W
            cy = H / 2.0
        else:
            cx, cy = W / 2.0, H / 2.0

        # Extend line to edges of world
        half_diag = math.hypot(W, H)
        p0 = np.array([cx - dx * half_diag, cy - dy * half_diag])
        p1 = np.array([cx + dx * half_diag, cy + dy * half_diag])

        # Clip to world bounds (rough)
        p0 = np.clip(p0, [0, 0], [W, H])
        p1 = np.clip(p1, [0, 0], [W, H])

        corridors.append({
            "name": corr.name,
            "p0": p0,
            "p1": p1,
            "sigma": corr.sigma * W,
        })

    return corridors


def _place_turbines_from_config(cfg: SiteConfig) -> np.ndarray:
    """Place turbines in world-space coordinates from config clusters."""
    W, H = cfg.simulation.agent.world_size
    rng = np.random.default_rng(cfg.layout_seed)
    n = cfg.turbine_count

    parts = []
    allocated = 0
    for cluster in cfg.clusters:
        count = int(n * cluster.fraction)
        allocated += count
        cx, cy = cluster.center
        sx, sy = cluster.spread
        pts = rng.normal(loc=[cx * W, cy * H], scale=[sx * W, sy * H], size=(count, 2))
        parts.append(pts)

    remaining = n - allocated
    if remaining > 0:
        bg = rng.uniform([0, 0], [W, H], size=(remaining, 2))
        parts.append(bg)

    turbines = np.vstack(parts)[:n]
    turbines[:, 0] = np.clip(turbines[:, 0], 0, W)
    turbines[:, 1] = np.clip(turbines[:, 1], 0, H)
    return turbines


def _spawn_migrants(
    n: int,
    corridors: List[Dict],
    speed: float,
    W: float, H: float,
    rng: np.random.Generator,
) -> List[Dict]:
    birds = []
    for _ in range(n):
        c = corridors[rng.integers(0, len(corridors))]
        p0, p1 = c["p0"], c["p1"]
        direction = p1 - p0
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        perp = np.array([-direction[1], direction[0]])

        lateral = rng.normal(0, c["sigma"])
        t = rng.uniform(0.0, 0.1)
        start = p0 * (1 - t) + p1 * t + perp * lateral + rng.normal(0, 1.5, size=2)
        start = np.clip(start, [0, 0], [W, H])

        end = p1 + perp * rng.normal(0, c["sigma"]) + rng.normal(0, 1.5, size=2)
        end = np.clip(end, [0, 0], [W, H])

        vec = end - start
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        s = speed * rng.uniform(0.75, 1.25)
        birds.append({"pos": start.copy(), "vel": vec * s, "type": "migrant", "alive": True})
    return birds


def _spawn_residents(
    n: int,
    speed: float,
    W: float, H: float,
    rng: np.random.Generator,
) -> List[Dict]:
    birds = []
    for _ in range(n):
        pos = rng.uniform([0, 0], [W, H])
        ang = rng.uniform(0, 2 * np.pi)
        s = speed * rng.uniform(0.6, 1.4)
        vel = np.array([np.cos(ang), np.sin(ang)]) * s
        birds.append({"pos": pos.copy(), "vel": vel, "type": "resident", "alive": True})
    return birds


def simulate_agent(cfg: SiteConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the agent-based collision simulation for 365 days.

    Returns:
        daily_birds: (365,) bird counts per day
        daily_deaths: (365,) death counts per day
        heat: (grid_n, grid_n) heatmap of strike locations
        turbines: (N, 2) turbine positions
    """
    agent = cfg.simulation.agent
    col = cfg.simulation.collision
    W, H = agent.world_size

    rng = np.random.default_rng(cfg.simulation.seed)
    corridors = _build_corridors_from_config(cfg)
    turbines = _place_turbines_from_config(cfg)
    month_to_season = build_month_to_season(cfg)

    DAYS = 365
    month_lengths = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                     7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    month = 1
    day_in_month = 1

    daily_deaths = np.zeros(DAYS, dtype=int)
    daily_birds = np.zeros(DAYS, dtype=int)

    grid_n = 60
    heat = np.zeros((grid_n, grid_n), dtype=float)

    for day in range(DAYS):
        season = month_to_season[month]

        birds_today = int(
            agent.birds_per_day_base
            * (0.35 + 0.65 * season.migration_intensity)
            * rng.uniform(0.85, 1.15)
        )
        daily_birds[day] = birds_today

        n_res = int(birds_today * season.resident_fraction)
        n_mig = birds_today - n_res

        birds = (
            _spawn_residents(n_res, agent.resident_speed, W, H, rng)
            + _spawn_migrants(n_mig, corridors, agent.migrant_speed, W, H, rng)
        )

        deaths = 0

        for _step in range(agent.steps_per_day):
            is_night = rng.random() < season.night_fraction

            for b in birds:
                if not b["alive"]:
                    continue

                b["pos"] = b["pos"] + b["vel"]
                x, y = b["pos"]

                if b["type"] == "resident":
                    if x < 0 or x > W:
                        b["vel"][0] *= -1
                        b["pos"][0] = np.clip(b["pos"][0], 0, W)
                    if y < 0 or y > H:
                        b["vel"][1] *= -1
                        b["pos"][1] = np.clip(b["pos"][1], 0, H)
                else:
                    if x < -5 or x > W + 5 or y < -5 or y > H + 5:
                        b["alive"] = False
                        continue

                d = np.linalg.norm(turbines - b["pos"], axis=1)
                min_i = int(np.argmin(d))
                inside = d[min_i] <= col.rotor_radius

                p_col = per_step_collision_prob(season, is_night, inside, col)
                if rng.random() < p_col:
                    deaths += 1
                    b["alive"] = False

                    gx = int(np.clip((turbines[min_i, 0] / W) * (grid_n - 1), 0, grid_n - 1))
                    gy = int(np.clip((turbines[min_i, 1] / H) * (grid_n - 1), 0, grid_n - 1))
                    heat[grid_n - 1 - gy, gx] += 1.0

        daily_deaths[day] = deaths

        day_in_month += 1
        if day_in_month > month_lengths[month]:
            day_in_month = 1
            month += 1
            if month == 13:
                month = 1

    return daily_birds, daily_deaths, heat, turbines


def run_agent_simulation(cfg: SiteConfig, out_dir: str = "outputs/agent-sim-outputs"):
    """
    Full agent-based simulation pipeline: simulate → plots → monthly summary.
    """
    from .charts import plot_agent_results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"agent_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running agent simulation for {cfg.site_name}...")
    print(f"  Turbines: {cfg.turbine_count}, Corridors: {len(cfg.corridors)}")

    daily_birds, daily_deaths, heat, turbines = simulate_agent(cfg)

    plot_agent_results(daily_birds, daily_deaths, heat, turbines, cfg, run_dir)

    # Monthly summary
    month_lengths_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    idx = 0
    print(f"\nMonthly totals ({cfg.site_name}):")
    for m, ml in enumerate(month_lengths_list):
        md = int(daily_deaths[idx:idx + ml].sum())
        mb = int(daily_birds[idx:idx + ml].sum())
        rate = md / max(mb, 1)
        print(f"  {month_names[m]}: deaths={md:5d}  birds={mb:7d}  rate={rate:.5f}")
        idx += ml

    print(f"\nOutput folder: {run_dir}")
    return run_dir
