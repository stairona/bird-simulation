#!/usr/bin/env python3
"""
simulate_isabella_bird_mortality.py

Creates a simulated 12-month bird mortality dataset for the Isabella Wind Project (Michigan)
and generates TWO updated bar charts:

1) Total Bird Mortality per Month — farm-wide
2) Average Bird Mortality per Turbine per Month — farm-wide

Updates requested:
- New unique output folder every run
- Larger fonts everywhere
- Cleaner white info box
- Only Mean + Peak shown in box

Outputs:
- out_sim_updated_YYYYMMDD_HHMMSS/
    - isabella_simulated_mortality.csv
    - graph_total_monthly_mortality_bar.png
    - graph_avg_per_turbine_monthly_bar.png

Requirements:
pip install numpy matplotlib
"""

import os
import csv
import math
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Create unique output folder
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"../outputs/simulation-outputs/out_sim_updated_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def compute_mode_int(values: np.ndarray) -> int:
    """Mode for integer arrays (returns smallest mode if tie)."""
    c = Counter(values.tolist())
    if not c:
        return 0
    max_freq = max(c.values())
    modes = [k for k, v in c.items() if v == max_freq]
    return int(min(modes))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------
# Ecologically-inspired field model
# (Gaussian corridors + curvature + turbine avoidance)
# ----------------------------
def make_turbine_layout(n_turbines: int, seed: int = 7):
    """
    Create a pseudo-spatial layout for 136 turbines across a rectangle.
    Not real coordinates; used to create spatial variability across ~56,000 acres.
    """
    rng = np.random.default_rng(seed)

    n1 = int(n_turbines * 0.45)
    n2 = int(n_turbines * 0.35)
    n3 = n_turbines - n1 - n2

    c1 = rng.normal(loc=(0.30, 0.65), scale=(0.08, 0.08), size=(n1, 2))
    c2 = rng.normal(loc=(0.70, 0.40), scale=(0.10, 0.07), size=(n2, 2))
    bg = rng.uniform(low=(0.10, 0.15), high=(0.90, 0.85), size=(n3, 2))
    xy = np.vstack([c1, c2, bg])

    xy[:, 0] = np.clip(xy[:, 0], 0.0, 1.0)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, 1.0)

    turbine_ids = np.arange(1, n_turbines + 1, dtype=int)
    return turbine_ids, xy

def gaussian2d(x, y, mx, my, sx, sy):
    return np.exp(-(((x - mx) ** 2) / (2 * sx ** 2) + ((y - my) ** 2) / (2 * sy ** 2)))

def corridor_density(xy: np.ndarray, month_idx: int):
    """
    Two dominant flyway components for Great Lakes region:
      - SW ↗ NE corridor
      - N ↕ S corridor
    With curvature to reflect bending around regional features.
    """
    x = xy[:, 0]
    y = xy[:, 1]

    t = month_idx / 11.0
    drift_x = 0.02 * math.sin(2 * math.pi * t)
    drift_y = 0.02 * math.cos(2 * math.pi * t)

    # SW–NE corridor
    theta = math.radians(35)
    xr = (x - 0.5) * math.cos(theta) - (y - 0.5) * math.sin(theta)
    yr = (x - 0.5) * math.sin(theta) + (y - 0.5) * math.cos(theta)

    curvature_strength = 0.06
    bend = curvature_strength * np.sin(2 * math.pi * (xr + 0.5) * 1.2)
    yr_curved = yr - bend

    swne = np.exp(-(yr_curved ** 2) / (2 * (0.10 ** 2)))

    # N–S corridor
    x_center = 0.52 + drift_x
    ns_curve = 0.03 * np.sin(2 * math.pi * (y + drift_y) * 1.1)
    ns = np.exp(-((x - (x_center + ns_curve)) ** 2) / (2 * (0.09 ** 2)))

    if month_idx in [3, 4]:  # Apr, May
        w_swne, w_ns = 0.45, 0.55
    elif month_idx in [8, 9]:  # Sep, Oct
        w_swne, w_ns = 0.60, 0.40
    else:
        w_swne, w_ns = 0.52, 0.48

    base = w_swne * swne + w_ns * ns

    blobs = (
        0.9 * gaussian2d(x, y, 0.62 + drift_x, 0.58 + drift_y, 0.10, 0.08) +
        0.7 * gaussian2d(x, y, 0.35 + drift_x, 0.42 + drift_y, 0.11, 0.09) +
        0.6 * gaussian2d(x, y, 0.78 + drift_x, 0.32 + drift_y, 0.12, 0.10)
    )
    return base + 0.35 * blobs

def turbine_avoidance_factor(xy: np.ndarray):
    """
    Simulate local avoidance around turbine clusters.
    """
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)

    knn = np.sort(d, axis=1)[:, :6]
    mean_knn = np.mean(knn, axis=1)

    z = (np.median(mean_knn) - mean_knn) / (np.std(mean_knn) + 1e-9)
    avoid = 0.65 * sigmoid(1.4 * z)
    return avoid

def migration_index_by_month():
    """
    0..1 migratory intensity index.
    """
    return np.array([0.05, 0.02, 0.25, 0.70, 0.80, 0.18, 0.12, 0.40, 0.65, 0.85, 0.28, 0.06], dtype=float)


# ----------------------------
# Simulation
# ----------------------------
def simulate_dataset(
    n_turbines: int = 136,
    seed: int = 7,
    base_rate: float = 0.75,
    winter_suppression: float = 0.35,
):
    """
    Returns rows: list of dicts with
      turbine_id, month_num, month, migration_index, mortality_count
    """
    rng = np.random.default_rng(seed)
    turbine_ids, xy = make_turbine_layout(n_turbines, seed=seed)
    avoid = turbine_avoidance_factor(xy)
    mig = migration_index_by_month()

    rows = []

    for m in range(12):
        m_name = MONTHS[m]
        m_idx = float(mig[m])

        dens = corridor_density(xy, m)
        dens = dens - dens.min()
        dens = dens / (dens.max() + 1e-12)

        if m in [0, 1, 11]:  # Jan, Feb, Dec
            season_factor = winter_suppression
        elif m == 10:  # Nov
            season_factor = 0.75
        else:
            season_factor = 1.0

        hetero = rng.lognormal(mean=0.0, sigma=0.30, size=n_turbines)
        lam = base_rate * m_idx * dens * (1.0 - avoid) * season_factor * hetero

        lam = 0.90 * lam
        mort = rng.poisson(lam=lam).astype(int)

        # February turbines 1–5 visited => force zero
        if m == 1:
            mort[0:5] = 0
            mort[5:] = np.minimum(mort[5:], 1)

        if m in [0, 11]:
            mort = np.minimum(mort, 1)

        for i, tid in enumerate(turbine_ids):
            rows.append({
                "turbine_id": int(tid),
                "month_num": int(m + 1),
                "month": m_name,
                "migration_index": float(m_idx),
                "mortality_count": int(mort[i]),
            })

    return rows


# ----------------------------
# Updated Graph 1
# Total Mortality Per Month
# ----------------------------
def plot_total_monthly_bar(rows, outpath_png):

    totals = np.zeros(12)

    for r in rows:
        totals[r["month_num"] - 1] += r["mortality_count"]

    mean_v = np.mean(totals)

    peak_idx = np.argmax(totals)
    peak_month = MONTHS[peak_idx]
    peak_val = int(totals[peak_idx])

    plt.figure(figsize=(16, 9), dpi=200)

    bars = plt.bar(MONTHS, totals, width=0.6)

    bars[peak_idx].set_color("red")

    plt.title(
        "Total Bird Mortality per Month — Isabella Wind Project (136 turbines)",
        fontsize=22,
        pad=20
    )

    plt.xlabel("Month", fontsize=18)
    plt.ylabel("Total Simulated Dead Birds (farm-wide)", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    info = (
        f"Mean monthly impact: {mean_v:.2f}\n"
        f"Peak month: {peak_month} ({peak_val})"
    )

    plt.gca().text(
        0.98,
        0.96,
        info,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=16,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.5"
        )
    )

    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.close()


# ----------------------------
# Updated Graph 2
# Average Mortality Per Turbine Per Month
# ----------------------------
def plot_avg_per_turbine_monthly_bar(rows, outpath_png):

    totals = np.zeros(12)

    for r in rows:
        totals[r["month_num"] - 1] += r["mortality_count"]

    avg = totals / 136

    mean_v = np.mean(avg)

    peak_idx = np.argmax(avg)
    peak_month = MONTHS[peak_idx]
    peak_val = avg[peak_idx]

    plt.figure(figsize=(16, 9), dpi=200)

    bars = plt.bar(MONTHS, avg, width=0.6)

    bars[peak_idx].set_color("red")

    plt.title(
        "Average Bird Mortality per Turbine per Month — Total Farm (136 turbines)",
        fontsize=22,
        pad=20
    )

    plt.xlabel("Month", fontsize=18)
    plt.ylabel("Average Dead Birds per Turbine", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    info = (
        f"Mean per turbine: {mean_v:.3f}\n"
        f"Peak month: {peak_month} ({peak_val:.3f})"
    )

    plt.gca().text(
        0.98,
        0.96,
        info,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=16,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.5"
        )
    )

    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.close()


# ----------------------------
# Export
# ----------------------------
def write_csv(rows, outpath_csv):
    fieldnames = ["turbine_id", "month_num", "month", "migration_index", "mortality_count"]
    with open(outpath_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def sanity_checks(rows):
    feb = [r for r in rows if r["month"] == "Feb"]
    feb_1_5 = [r for r in feb if 1 <= r["turbine_id"] <= 5]
    assert all(r["mortality_count"] == 0 for r in feb_1_5), "Feb turbines 1..5 must be 0"

    totals = {m: 0 for m in MONTHS}
    for r in rows:
        totals[r["month"]] += r["mortality_count"]

    return totals


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--base_rate", type=float, default=0.75, help="Base collision scale")
    args = ap.parse_args()

    ensure_dir(OUTPUT_DIR)

    rows = simulate_dataset(
        n_turbines=136,
        seed=args.seed,
        base_rate=args.base_rate,
        winter_suppression=0.35
    )

    totals = sanity_checks(rows)

    out_csv = os.path.join(OUTPUT_DIR, "isabella_simulated_mortality.csv")
    write_csv(rows, out_csv)

    out_png1 = os.path.join(OUTPUT_DIR, "graph_total_monthly_mortality_bar.png")
    out_png2 = os.path.join(OUTPUT_DIR, "graph_avg_per_turbine_monthly_bar.png")

    plot_total_monthly_bar(rows, out_png1)
    plot_avg_per_turbine_monthly_bar(rows, out_png2)

    print("Saved:", out_csv)
    print("Saved:", out_png1)
    print("Saved:", out_png2)
    print("Monthly totals:", totals)
    print("Migration index (0..1):", dict(zip(MONTHS, migration_index_by_month().round(2).tolist())))
    print("New output folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()