"""Tests for Phase 2 chart generation."""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.phase2_mortality.charts import (
    plot_agent_results,
    plot_avg_per_turbine_monthly_bar,
    plot_total_monthly_bar,
)
from src.core.calendar import MONTH_NAMES


@pytest.fixture
def sample_rows(isabella_cfg):
    """Minimal simulation rows for chart tests."""
    rows = []
    for m in range(12):
        for tid in range(isabella_cfg.turbine_count):
            rows.append({
                "turbine_id": tid,
                "month_num": m + 1,
                "month": MONTH_NAMES[m],
                "migration_index": 0.5,
                "mortality_count": int(np.random.default_rng(42 + m + tid).integers(0, 5)),
            })
    return rows


class TestStatCharts:
    def test_total_monthly_bar_creates_png(self, sample_rows, isabella_cfg, tmp_path):
        out = str(tmp_path / "total.png")
        plot_total_monthly_bar(sample_rows, out, isabella_cfg)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_avg_per_turbine_bar_creates_png(self, sample_rows, isabella_cfg, tmp_path):
        out = str(tmp_path / "avg.png")
        plot_avg_per_turbine_monthly_bar(sample_rows, out, isabella_cfg)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100


class TestAgentCharts:
    def test_plot_agent_results_creates_files(self, isabella_cfg, tmp_path):
        rng = np.random.default_rng(42)
        daily_birds = rng.integers(100, 600, size=365)
        daily_deaths = rng.integers(0, 10, size=365)
        heat = rng.random((60, 60))
        W, H = isabella_cfg.simulation.agent.world_size
        turbines = rng.uniform(0, 1, size=(isabella_cfg.turbine_count, 2))
        turbines[:, 0] *= W
        turbines[:, 1] *= H

        out_dir = str(tmp_path / "agent_plots")
        os.makedirs(out_dir)
        plot_agent_results(daily_birds, daily_deaths, heat, turbines, isabella_cfg, out_dir)

        expected = [
            "daily_fatalities.png",
            "daily_fatality_rate.png",
            "strike_heatmap.png",
            "layout_corridors.png",
        ]
        for fname in expected:
            fpath = os.path.join(out_dir, fname)
            assert os.path.exists(fpath), f"Missing {fname}"
            assert os.path.getsize(fpath) > 100
