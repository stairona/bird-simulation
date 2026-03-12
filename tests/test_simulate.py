"""Tests for the Poisson mortality simulation."""

import copy

import numpy as np
import pytest

from src.phase2_mortality.simulate import simulate_dataset, monthly_totals, MONTH_NAMES


class TestSimulateDataset:
    def test_produces_rows(self, isabella_cfg):
        rows = simulate_dataset(isabella_cfg)
        assert len(rows) > 0

    def test_12_months_per_turbine(self, isabella_cfg):
        rows = simulate_dataset(isabella_cfg)
        # 136 turbines x 12 months = 1632 rows
        assert len(rows) == 136 * 12

    def test_row_schema(self, isabella_cfg):
        rows = simulate_dataset(isabella_cfg)
        r = rows[0]
        assert "turbine_id" in r
        assert "month_num" in r
        assert "month" in r
        assert "migration_index" in r
        assert "mortality_count" in r

    def test_mortality_nonnegative(self, isabella_cfg):
        rows = simulate_dataset(isabella_cfg)
        for r in rows:
            assert r["mortality_count"] >= 0

    def test_deterministic_with_same_seed(self, isabella_cfg):
        rows1 = simulate_dataset(isabella_cfg)
        rows2 = simulate_dataset(isabella_cfg)
        m1 = [r["mortality_count"] for r in rows1]
        m2 = [r["mortality_count"] for r in rows2]
        assert m1 == m2


class TestAvoidanceBugFix:
    """Verify that collision.avoidance actually affects the Poisson model."""

    def test_higher_avoidance_means_fewer_deaths(self, isabella_cfg):
        cfg_low = copy.deepcopy(isabella_cfg)
        cfg_low.simulation.collision.avoidance = 0.50

        cfg_high = copy.deepcopy(isabella_cfg)
        cfg_high.simulation.collision.avoidance = 0.95

        rows_low = simulate_dataset(cfg_low)
        rows_high = simulate_dataset(cfg_high)

        total_low = sum(r["mortality_count"] for r in rows_low)
        total_high = sum(r["mortality_count"] for r in rows_high)

        assert total_low > total_high, (
            f"Higher avoidance should mean fewer deaths: "
            f"avoidance=0.50 gave {total_low}, avoidance=0.95 gave {total_high}"
        )

    def test_different_avoidance_produces_different_results(self, isabella_cfg):
        cfg1 = copy.deepcopy(isabella_cfg)
        cfg1.simulation.collision.avoidance = 0.60

        cfg2 = copy.deepcopy(isabella_cfg)
        cfg2.simulation.collision.avoidance = 0.90

        total1 = sum(r["mortality_count"] for r in simulate_dataset(cfg1))
        total2 = sum(r["mortality_count"] for r in simulate_dataset(cfg2))

        assert total1 != total2


class TestBaseRateSweep:
    """Verify that base_rate affects mortality."""

    def test_higher_base_rate_more_deaths(self, isabella_cfg):
        cfg_low = copy.deepcopy(isabella_cfg)
        cfg_low.simulation.base_rate = 0.25

        cfg_high = copy.deepcopy(isabella_cfg)
        cfg_high.simulation.base_rate = 1.50

        total_low = sum(r["mortality_count"] for r in simulate_dataset(cfg_low))
        total_high = sum(r["mortality_count"] for r in simulate_dataset(cfg_high))

        assert total_high > total_low


class TestMonthlyTotals:
    def test_12_months(self, isabella_cfg):
        rows = simulate_dataset(isabella_cfg)
        totals = monthly_totals(rows)
        assert len(totals) == 12
        assert set(totals.keys()) == set(MONTH_NAMES)

    def test_totals_are_nonnegative(self, isabella_cfg):
        rows = simulate_dataset(isabella_cfg)
        totals = monthly_totals(rows)
        for v in totals.values():
            assert v >= 0
