"""Tests for config loading and validation."""

import os
import tempfile

import pytest

from src.core.config import load_config, SiteConfig


class TestLoadConfig:
    def test_isabella_loads(self, isabella_cfg):
        assert isinstance(isabella_cfg, SiteConfig)
        assert isabella_cfg.site_name == "Isabella Wind Project"
        assert isabella_cfg.turbine_count == 136

    def test_isabella_has_12_months(self, isabella_cfg):
        assert len(isabella_cfg.monthly_calendar) == 12

    def test_isabella_has_seasons(self, isabella_cfg):
        assert set(isabella_cfg.seasons.keys()) == {"winter", "spring", "summer", "fall"}

    def test_isabella_has_corridors(self, isabella_cfg):
        assert len(isabella_cfg.corridors) == 2
        assert isabella_cfg.corridors[0].name == "SW-NE flyway"

    def test_isabella_has_species(self, isabella_cfg):
        assert "songbirds" in isabella_cfg.species
        assert "raptors" in isabella_cfg.species
        assert "local" in isabella_cfg.species

    def test_isabella_simulation_params(self, isabella_cfg):
        assert isabella_cfg.simulation.seed == 7
        assert isabella_cfg.simulation.base_rate == 0.75
        assert isabella_cfg.simulation.collision.avoidance == 0.80

    def test_migratory_species_excludes_local(self, isabella_cfg):
        keys = isabella_cfg.migratory_species_keys
        assert "local" not in keys
        assert "songbirds" in keys

    def test_winter_months(self, isabella_cfg):
        assert isabella_cfg.winter_months == {12, 1, 2}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_short_calendar_raises(self, tmp_path):
        """A config with fewer than 12 calendar entries should fail."""
        yaml_content = """
site:
  name: "Bad Config"
  region: "Test"
turbines:
  count: 10
  layout_seed: 1
  clusters: []
corridors: []
species: {}
monthly_calendar:
  - month: "Jan"
    migration_index: 0.1
    intensity: 0.1
    label: "test"
    season: "winter"
seasons: {}
maps: {}
simulation: {}
"""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="monthly_calendar must have exactly 12"):
            load_config(str(cfg_path))
