"""
Sensitivity sweep tool.

Varies a single simulation parameter across a range of values and
records how annual mortality changes. Produces a summary CSV and chart.

Supported sweep parameters:
  - avoidance (collision.avoidance): 0.0 to 0.99
  - base_rate (simulation.base_rate): 0.1 to 5.0
  - turbine_count (turbines.count): 10 to 500
  - base_strike_prob (collision.base_strike_prob): 0.0001 to 0.01
"""

from __future__ import annotations

import copy
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.config import SiteConfig, load_config
from ..phase2_mortality.simulate import MONTH_NAMES, monthly_totals, simulate_dataset

SWEEPABLE_PARAMS = {
    "avoidance": {
        "path": ("simulation", "collision", "avoidance"),
        "default_range": (0.0, 0.95, 10),
        "label": "Collision Avoidance Rate",
    },
    "base_rate": {
        "path": ("simulation", "base_rate"),
        "default_range": (0.2, 2.0, 10),
        "label": "Base Mortality Rate",
    },
    "turbine_count": {
        "path": ("turbines", "count"),
        "default_range": (20, 200, 10),
        "label": "Number of Turbines",
        "dtype": int,
    },
    "base_strike_prob": {
        "path": ("simulation", "collision", "base_strike_prob"),
        "default_range": (0.0005, 0.008, 10),
        "label": "Base Strike Probability",
    },
}


def _set_nested(raw: dict, keys: Tuple[str, ...], value):
    """Set a value in a nested dict by key path."""
    # Ensure native Python types for YAML serialization
    if isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    for k in keys[:-1]:
        raw = raw[k]
    raw[keys[-1]] = value


def _get_nested(raw: dict, keys: Tuple[str, ...]):
    """Get a value from a nested dict by key path."""
    for k in keys:
        raw = raw[k]
    return raw


def run_sweep(
    config_path: str,
    param: str,
    values: Optional[List[float]] = None,
    n_steps: int = 10,
    out_dir: str = "outputs/sweep",
) -> str:
    """
    Sweep a parameter and record mortality for each value.

    Args:
        config_path: path to the base YAML config
        param: parameter key from SWEEPABLE_PARAMS
        values: explicit list of values (overrides n_steps)
        n_steps: number of evenly spaced steps if values not given
        out_dir: output directory

    Returns:
        Path to the output directory.
    """
    if param not in SWEEPABLE_PARAMS:
        available = ", ".join(SWEEPABLE_PARAMS.keys())
        raise ValueError(f"Unknown sweep param '{param}'. Available: {available}")

    spec = SWEEPABLE_PARAMS[param]
    key_path = spec["path"]
    dtype = spec.get("dtype", float)

    if values is None:
        lo, hi, default_n = spec["default_range"]
        values = [float(v) for v in np.linspace(lo, hi, n_steps if n_steps else default_n)]

    if dtype is int:
        values = [int(round(v)) for v in values]
    else:
        values = [float(v) for v in values]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"sweep_{param}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    import yaml
    with open(config_path) as f:
        base_raw = yaml.safe_load(f)

    original_value = _get_nested(base_raw, key_path)
    print(f"Sweeping '{param}' ({spec['label']})")
    print(f"  Current value: {original_value}")
    print(f"  Sweep values: {values}")

    results: List[Dict] = []

    for val in values:
        raw_copy = copy.deepcopy(base_raw)
        _set_nested(raw_copy, key_path, val)

        tmp_path = os.path.join(run_dir, "_tmp_sweep.yaml")
        with open(tmp_path, "w") as f:
            yaml.dump(raw_copy, f, default_flow_style=False, sort_keys=False)

        cfg = load_config(tmp_path)
        rows = simulate_dataset(cfg)
        totals = monthly_totals(rows)
        annual = sum(totals.values())

        entry = {"param_value": val, "annual_mortality": annual}
        for m in MONTH_NAMES:
            entry[m] = totals[m]
        results.append(entry)

        print(f"  {param}={val}: annual mortality = {annual}")

    # Cleanup temp file
    tmp_path = os.path.join(run_dir, "_tmp_sweep.yaml")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    _write_sweep_csv(results, param, run_dir)
    _plot_sweep(results, param, spec["label"], run_dir)

    print(f"Sweep output: {run_dir}")
    return run_dir


def _write_sweep_csv(results: List[Dict], param: str, run_dir: str):
    path = os.path.join(run_dir, f"sweep_{param}.csv")
    fieldnames = ["param_value", "annual_mortality"] + list(MONTH_NAMES)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved: {path}")


def _plot_sweep(results: List[Dict], param: str, label: str, run_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["param_value"] for r in results]
    y = [r["annual_mortality"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(x, y, alpha=0.15, color="#2196F3")
    ax.set_xlabel(label)
    ax.set_ylabel("Estimated Annual Mortality")
    ax.set_title(f"Sensitivity Sweep — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(run_dir, f"sweep_{param}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
