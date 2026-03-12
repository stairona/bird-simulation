"""
Scenario comparison tool.

Runs the mortality simulation for multiple config files (or config
variants) and produces side-by-side comparison charts and a summary CSV.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ..core.config import SiteConfig, load_config
from ..phase2_mortality.simulate import MONTH_NAMES, monthly_totals, simulate_dataset


class ScenarioResult:
    __slots__ = ("name", "config_path", "rows", "totals", "annual_total")

    def __init__(self, name: str, config_path: str, rows: List[Dict],
                 totals: Dict[str, int], annual_total: int):
        self.name = name
        self.config_path = config_path
        self.rows = rows
        self.totals = totals
        self.annual_total = annual_total


def _make_result(name: str, config_path: str, cfg: SiteConfig) -> ScenarioResult:
    rows = simulate_dataset(cfg)
    totals = monthly_totals(rows)
    annual = sum(totals.values())
    return ScenarioResult(
        name=name, config_path=config_path,
        rows=rows, totals=totals, annual_total=annual,
    )


def compare_scenarios(
    config_paths: List[str],
    names: Optional[List[str]] = None,
    out_dir: str = "outputs/comparison",
) -> str:
    """
    Run mortality simulations for each config and compare results.

    Args:
        config_paths: list of YAML config file paths
        names: optional labels for each scenario (defaults to site names)
        out_dir: output directory for comparison artifacts

    Returns:
        Path to the output directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"compare_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    results: List[ScenarioResult] = []
    for i, path in enumerate(config_paths):
        cfg = load_config(path)
        name = names[i] if names and i < len(names) else cfg.site_name
        result = _make_result(name, path, cfg)
        results.append(result)
        print(f"  Scenario '{name}': {result.annual_total} annual mortality")

    _write_summary_csv(results, run_dir)
    _plot_comparison_bar(results, run_dir)
    _plot_comparison_monthly(results, run_dir)

    print(f"Comparison output: {run_dir}")
    return run_dir


def _write_summary_csv(results: List[ScenarioResult], run_dir: str):
    path = os.path.join(run_dir, "comparison_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Scenario", "Annual Total"] + list(MONTH_NAMES)
        writer.writerow(header)
        for r in results:
            row = [r.name, r.annual_total] + [r.totals[m] for m in MONTH_NAMES]
            writer.writerow(row)
    print(f"Saved: {path}")


def _plot_comparison_bar(results: List[ScenarioResult], run_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    totals = [r.annual_total for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 2), 6))
    bars = ax.bar(range(len(results)), totals, color="#2196F3", edgecolor="white")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Estimated Annual Mortality")
    ax.set_title("Scenario Comparison — Annual Mortality")

    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    path = os.path.join(run_dir, "comparison_annual_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_comparison_monthly(results: List[ScenarioResult], run_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(12)
    width = 0.8 / max(len(results), 1)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]

    for i, r in enumerate(results):
        vals = [r.totals[m] for m in MONTH_NAMES]
        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=r.name,
               color=colors[i % len(colors)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("Monthly Mortality")
    ax.set_title("Scenario Comparison — Monthly Mortality")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(run_dir, "comparison_monthly_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
