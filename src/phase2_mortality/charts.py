"""
Chart generation for Phase 2 mortality outputs.

Produces publication-ready bar charts and heatmaps from simulation data.
All titles and labels use the site name from config.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from ..core.config import SiteConfig

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Statistical simulation charts ─────────────────────────────────

def plot_total_monthly_bar(rows: List[Dict], outpath: str, cfg: SiteConfig):
    """Bar chart: total mortality per month (farm-wide)."""
    totals = np.zeros(12)
    for r in rows:
        totals[r["month_num"] - 1] += r["mortality_count"]

    mean_v = np.mean(totals)
    peak_idx = int(np.argmax(totals))
    peak_month = MONTH_NAMES[peak_idx]
    peak_val = int(totals[peak_idx])

    plt.figure(figsize=(16, 9), dpi=200)
    bars = plt.bar(MONTH_NAMES, totals, width=0.6)
    bars[peak_idx].set_color("red")

    plt.title(
        f"Total Bird Mortality per Month — {cfg.site_name} ({cfg.turbine_count} turbines)",
        fontsize=22, pad=20,
    )
    plt.xlabel("Month", fontsize=18)
    plt.ylabel("Total Simulated Dead Birds (farm-wide)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    info = f"Mean monthly impact: {mean_v:.2f}\nPeak month: {peak_month} ({peak_val})"
    plt.gca().text(
        0.98, 0.96, info,
        transform=plt.gca().transAxes, ha="right", va="top", fontsize=16,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_avg_per_turbine_monthly_bar(rows: List[Dict], outpath: str, cfg: SiteConfig):
    """Bar chart: average mortality per turbine per month."""
    totals = np.zeros(12)
    for r in rows:
        totals[r["month_num"] - 1] += r["mortality_count"]

    avg = totals / cfg.turbine_count
    mean_v = np.mean(avg)
    peak_idx = int(np.argmax(avg))
    peak_month = MONTH_NAMES[peak_idx]
    peak_val = avg[peak_idx]

    plt.figure(figsize=(16, 9), dpi=200)
    bars = plt.bar(MONTH_NAMES, avg, width=0.6)
    bars[peak_idx].set_color("red")

    plt.title(
        f"Average Bird Mortality per Turbine per Month — {cfg.site_name} ({cfg.turbine_count} turbines)",
        fontsize=22, pad=20,
    )
    plt.xlabel("Month", fontsize=18)
    plt.ylabel("Average Dead Birds per Turbine", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    info = f"Mean per turbine: {mean_v:.3f}\nPeak month: {peak_month} ({peak_val:.3f})"
    plt.gca().text(
        0.98, 0.96, info,
        transform=plt.gca().transAxes, ha="right", va="top", fontsize=16,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ── Agent simulation charts ───────────────────────────────────────

def plot_agent_results(
    daily_birds: np.ndarray,
    daily_deaths: np.ndarray,
    heat: np.ndarray,
    turbines: np.ndarray,
    cfg: SiteConfig,
    out_dir: str,
):
    """Generate all plots from the agent-based simulation."""
    days = np.arange(1, len(daily_deaths) + 1)
    death_rate = daily_deaths / np.maximum(daily_birds, 1)

    # Daily fatalities
    plt.figure(figsize=(11, 6))
    plt.plot(days, daily_deaths, linewidth=1.2)
    plt.title(f"Simulated Daily Bird Fatalities — {cfg.site_name}")
    plt.xlabel("Day of Year")
    plt.ylabel("Deaths (count)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_fatalities.png"), dpi=200)
    plt.close()

    # Daily fatality rate
    plt.figure(figsize=(11, 6))
    plt.plot(days, death_rate, linewidth=1.2)
    plt.title(f"Simulated Daily Fatality Rate — {cfg.site_name}")
    plt.xlabel("Day of Year")
    plt.ylabel("Rate (Deaths / Birds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_fatality_rate.png"), dpi=200)
    plt.close()

    # Strike heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(heat, aspect="auto")
    plt.title(f"Turbine Strike Hotspots — {cfg.site_name}")
    plt.xlabel("X (west→east)")
    plt.ylabel("Y (south→north)")
    plt.colorbar(label="Strike count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strike_heatmap.png"), dpi=200)
    plt.close()

    # Turbine layout + corridors
    W, H = cfg.simulation.agent.world_size
    plt.figure(figsize=(7, 7))
    plt.scatter(turbines[:, 0], turbines[:, 1], s=18, label="Turbines")

    import math
    for corr in cfg.corridors:
        theta = math.radians(corr.angle_deg)
        dx, dy = math.cos(theta), math.sin(theta)
        if corr.center is not None:
            cx, cy = corr.center[0] * W, corr.center[1] * H
        elif corr.center_x is not None:
            cx, cy = corr.center_x * W, H / 2.0
        else:
            cx, cy = W / 2.0, H / 2.0

        half = math.hypot(W, H) * 0.6
        x0 = np.clip(cx - dx * half, 0, W)
        y0 = np.clip(cy - dy * half, 0, H)
        x1 = np.clip(cx + dx * half, 0, W)
        y1 = np.clip(cy + dy * half, 0, H)
        plt.plot([x0, x1], [y0, y1], linewidth=2, label=corr.name)

    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.title(f"Turbine Layout + Migration Corridors — {cfg.site_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "layout_corridors.png"), dpi=200)
    plt.close()

    print(f"Saved 4 plots to {out_dir}/")
