"""Tests for src.tools.sweep — sensitivity sweep tool."""

import os

import pytest

from src.tools.sweep import SWEEPABLE_PARAMS, run_sweep


ISABELLA_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "isabella.yaml",
)


class TestSweepableParams:
    def test_known_params(self):
        expected = {"avoidance", "base_rate", "turbine_count", "base_strike_prob"}
        assert expected == set(SWEEPABLE_PARAMS.keys())

    def test_params_have_required_fields(self):
        for key, spec in SWEEPABLE_PARAMS.items():
            assert "path" in spec
            assert "default_range" in spec
            assert "label" in spec
            lo, hi, n = spec["default_range"]
            assert lo < hi
            assert n >= 2


class TestRunSweep:
    def test_avoidance_sweep(self, tmp_path):
        out = run_sweep(
            config_path=ISABELLA_CONFIG,
            param="avoidance",
            values=[0.5, 0.8, 0.95],
            out_dir=str(tmp_path),
        )
        assert os.path.isdir(out)
        files = os.listdir(out)
        assert any("sweep_avoidance.csv" in f for f in files)
        assert any("sweep_avoidance.png" in f for f in files)

    def test_base_rate_sweep(self, tmp_path):
        out = run_sweep(
            config_path=ISABELLA_CONFIG,
            param="base_rate",
            values=[0.5, 1.0],
            out_dir=str(tmp_path),
        )
        assert os.path.isdir(out)

    def test_invalid_param_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown sweep param"):
            run_sweep(
                config_path=ISABELLA_CONFIG,
                param="nonexistent",
                out_dir=str(tmp_path),
            )

    def test_higher_avoidance_fewer_deaths(self, tmp_path):
        out = run_sweep(
            config_path=ISABELLA_CONFIG,
            param="avoidance",
            values=[0.0, 0.5, 0.95],
            out_dir=str(tmp_path),
        )
        import csv as csv_mod
        csv_path = None
        for f in os.listdir(out):
            if f.endswith(".csv"):
                csv_path = os.path.join(out, f)
                break
        assert csv_path is not None

        with open(csv_path) as f:
            rows = list(csv_mod.DictReader(f))
        mortalities = [int(r["annual_mortality"]) for r in rows]
        # Higher avoidance should mean fewer deaths
        assert mortalities[0] >= mortalities[-1]
