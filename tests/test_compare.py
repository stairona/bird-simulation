"""Tests for src.tools.compare — scenario comparison tool."""

import os
import pytest

from src.tools.compare import ScenarioResult, compare_scenarios


ISABELLA_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "isabella.yaml",
)


class TestScenarioResult:
    def test_construction(self):
        r = ScenarioResult(
            name="test", config_path="/tmp/test.yaml",
            rows=[{"month": "Jan", "mortality_count": 5}],
            totals={"Jan": 5}, annual_total=5,
        )
        assert r.name == "test"
        assert r.annual_total == 5


class TestCompareScenarios:
    def test_single_scenario(self, tmp_path):
        out = compare_scenarios(
            config_paths=[ISABELLA_CONFIG],
            names=["Isabella"],
            out_dir=str(tmp_path),
        )
        assert os.path.isdir(out)
        files = os.listdir(out)
        assert any("summary" in f for f in files)
        assert any("annual_bar" in f for f in files)
        assert any("monthly_bar" in f for f in files)

    def test_two_scenarios(self, tmp_path):
        out = compare_scenarios(
            config_paths=[ISABELLA_CONFIG, ISABELLA_CONFIG],
            names=["Scenario A", "Scenario B"],
            out_dir=str(tmp_path),
        )
        csv_files = [f for f in os.listdir(out) if f.endswith(".csv")]
        assert len(csv_files) >= 1
