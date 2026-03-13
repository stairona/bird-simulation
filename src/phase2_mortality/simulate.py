"""
Phase 2 — Statistical (Poisson) mortality simulation.

Generalized from simulate_isabella_bird_mortality.py.
All site-specific values (turbine count, corridors, migration calendar,
density blobs) come from the SiteConfig YAML.

Pipeline:
  1. Generate turbine layout from config clusters
  2. Compute corridor density per turbine per month
  3. Apply Poisson mortality with avoidance, season, and heterogeneity
  4. Return row-level data for charting
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List

import numpy as np

from ..core.config import SiteConfig
from ..core.corridors import corridor_density
from ..core.calendar import migration_index_array
from ..core.turbines import make_turbine_layout, turbine_avoidance_factor

from ..core.calendar import MONTH_NAMES


def simulate_dataset(cfg: SiteConfig) -> List[Dict]:
    """
    Run the Poisson mortality model for all turbines x 12 months.

    Returns list of dicts with keys:
        turbine_id, month_num, month, migration_index, mortality_count
    """
    sim = cfg.simulation
    rng = np.random.default_rng(sim.seed)

    turbine_ids, xy = make_turbine_layout(cfg)
    avoid = turbine_avoidance_factor(xy)
    mig = migration_index_array(cfg)
    n = cfg.turbine_count

    winter_month_indices = {m - 1 for m in cfg.winter_months}

    rows: List[Dict] = []

    for m in range(12):
        m_name = MONTH_NAMES[m]
        m_idx = float(mig[m])

        dens = corridor_density(xy, m, cfg)
        dens = dens - dens.min()
        dens = dens / (dens.max() + 1e-12)

        if m in winter_month_indices:
            season_factor = sim.winter_suppression
        else:
            season_factor = 1.0

        hetero = rng.lognormal(mean=0.0, sigma=sim.heterogeneity_sigma, size=n)
        config_avoid = sim.collision.avoidance
        lam = (sim.base_rate * m_idx * dens * (1.0 - avoid)
               * (1.0 - config_avoid) * season_factor * hetero)
        lam = sim.mortality_scaling * lam
        mort = rng.poisson(lam=lam).astype(int)

        if m in winter_month_indices:
            mort = np.minimum(mort, sim.winter_cap)

        for i, tid in enumerate(turbine_ids):
            rows.append({
                "turbine_id": int(tid),
                "month_num": int(m + 1),
                "month": m_name,
                "migration_index": float(m_idx),
                "mortality_count": int(mort[i]),
            })

    return rows


def write_csv(rows: List[Dict], outpath: str):
    """Write simulation rows to CSV."""
    fieldnames = ["turbine_id", "month_num", "month", "migration_index", "mortality_count"]
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def monthly_totals(rows: List[Dict]) -> Dict[str, int]:
    """Aggregate total mortality per month."""
    totals = {m: 0 for m in MONTH_NAMES}
    for r in rows:
        totals[r["month"]] += r["mortality_count"]
    return totals


def run_simulation(cfg: SiteConfig, out_dir: str = "outputs/simulation-outputs"):
    """
    Full Phase 2 statistical pipeline: simulate → CSV → charts.

    Creates a timestamped subfolder with all outputs.
    """
    from .charts import plot_total_monthly_bar, plot_avg_per_turbine_monthly_bar

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"sim_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rows = simulate_dataset(cfg)
    totals = monthly_totals(rows)

    csv_path = os.path.join(run_dir, "simulated_mortality.csv")
    write_csv(rows, csv_path)

    png1 = os.path.join(run_dir, "graph_total_monthly_mortality_bar.png")
    png2 = os.path.join(run_dir, "graph_avg_per_turbine_monthly_bar.png")

    plot_total_monthly_bar(rows, png1, cfg)
    plot_avg_per_turbine_monthly_bar(rows, png2, cfg)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png1}")
    print(f"Saved: {png2}")
    print(f"Monthly totals: {totals}")
    print(f"Migration index: {dict(zip(MONTH_NAMES, migration_index_array(cfg).round(2).tolist()))}")
    print(f"Output folder: {run_dir}")

    return run_dir, rows
