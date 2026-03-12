"""Tests for src.tools.generate_config — auto config generation."""

import os
import tempfile

import numpy as np
import pytest

from src.tools.generate_config import (
    _derive_clusters,
    generate_config,
    load_turbine_csv,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small turbine CSV for testing."""
    csv_path = tmp_path / "turbines.csv"
    csv_path.write_text(
        "latitude,longitude\n"
        "43.745,-84.701\n"
        "43.748,-84.695\n"
        "43.752,-84.690\n"
        "43.740,-84.710\n"
        "43.755,-84.685\n"
        "43.738,-84.715\n"
        "43.760,-84.680\n"
        "43.735,-84.720\n"
        "43.750,-84.698\n"
        "43.742,-84.705\n"
    )
    return str(csv_path)


@pytest.fixture
def alt_csv(tmp_path):
    """CSV with alternative column names (lat/lon)."""
    csv_path = tmp_path / "alt_turbines.csv"
    csv_path.write_text(
        "id,lat,lon,name\n"
        "1,43.745,-84.701,T1\n"
        "2,43.748,-84.695,T2\n"
        "3,43.752,-84.690,T3\n"
    )
    return str(csv_path)


class TestLoadTurbineCSV:
    def test_standard_columns(self, sample_csv):
        lats, lons = load_turbine_csv(sample_csv)
        assert len(lats) == 10
        assert len(lons) == 10
        assert lats[0] == pytest.approx(43.745)
        assert lons[0] == pytest.approx(-84.701)

    def test_alt_columns(self, alt_csv):
        lats, lons = load_turbine_csv(alt_csv)
        assert len(lats) == 3

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_turbine_csv("/nonexistent/path.csv")

    def test_bad_columns(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("x,y\n1,2\n")
        with pytest.raises(ValueError, match="latitude"):
            load_turbine_csv(str(bad))

    def test_empty_csv(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("latitude,longitude\n")
        with pytest.raises(ValueError, match="No valid rows"):
            load_turbine_csv(str(empty))


class TestDeriveClusters:
    def test_single_cluster_for_small_dataset(self):
        xy = np.random.default_rng(42).uniform(0, 1, size=(5, 2))
        clusters = _derive_clusters(xy, max_clusters=4)
        assert len(clusters) == 1
        assert clusters[0]["fraction"] == 1.0

    def test_multiple_clusters(self):
        rng = np.random.default_rng(42)
        group1 = rng.normal(loc=[0.2, 0.3], scale=0.05, size=(30, 2))
        group2 = rng.normal(loc=[0.8, 0.7], scale=0.05, size=(30, 2))
        xy = np.vstack([group1, group2])
        clusters = _derive_clusters(xy, max_clusters=4)
        assert len(clusters) >= 2
        total_frac = sum(c["fraction"] for c in clusters)
        assert total_frac == pytest.approx(1.0, abs=0.05)

    def test_cluster_centers_in_range(self):
        xy = np.random.default_rng(42).uniform(0, 1, size=(40, 2))
        clusters = _derive_clusters(xy)
        for c in clusters:
            assert 0 <= c["center"][0] <= 1
            assert 0 <= c["center"][1] <= 1


class TestGenerateConfig:
    def test_produces_valid_yaml(self, sample_csv, tmp_path):
        out = generate_config(
            turbine_csv=sample_csv,
            region="atlantic",
            site_name="Test Farm",
            output_path=str(tmp_path / "test.yaml"),
        )
        assert os.path.exists(out)

        from src.core.config import load_config
        cfg = load_config(out)
        assert cfg.site_name == "Test Farm"
        assert cfg.turbine_count == 10
        assert len(cfg.corridors) >= 2
        assert len(cfg.monthly_calendar) == 12

    def test_default_name_from_csv(self, sample_csv, tmp_path):
        out = generate_config(
            turbine_csv=sample_csv,
            region="mississippi",
            output_path=str(tmp_path / "auto.yaml"),
        )
        from src.core.config import load_config
        cfg = load_config(out)
        assert "Turbines" in cfg.site_name or len(cfg.site_name) > 0

    def test_all_flyways(self, sample_csv, tmp_path):
        for region in ["atlantic", "mississippi", "central", "pacific",
                        "western_palearctic", "east_asian"]:
            out = generate_config(
                turbine_csv=sample_csv,
                region=region,
                output_path=str(tmp_path / f"{region}.yaml"),
            )
            from src.core.config import load_config
            cfg = load_config(out)
            assert len(cfg.monthly_calendar) == 12
