"""Tests for src.core.flyways — flyway preset definitions."""

import pytest

from src.core.flyways import (
    FLYWAY_PRESETS,
    SEASON_DEFAULTS,
    available_flyways,
    get_flyway,
)


class TestAvailableFlyways:
    def test_returns_dict(self):
        fw = available_flyways()
        assert isinstance(fw, dict)
        assert len(fw) >= 6

    def test_all_keys_present(self):
        expected = {"atlantic", "mississippi", "central", "pacific",
                    "western_palearctic", "east_asian"}
        assert expected.issubset(available_flyways().keys())

    def test_labels_are_strings(self):
        for key, label in available_flyways().items():
            assert isinstance(label, str)
            assert len(label) > 5


class TestGetFlyway:
    def test_valid_flyway(self):
        fw = get_flyway("atlantic")
        assert "corridors" in fw
        assert "species" in fw
        assert "monthly_calendar" in fw

    def test_invalid_flyway_raises(self):
        with pytest.raises(KeyError, match="Unknown flyway"):
            get_flyway("nonexistent")


class TestFlywayPresetStructure:
    @pytest.fixture(params=list(FLYWAY_PRESETS.keys()))
    def flyway(self, request):
        return FLYWAY_PRESETS[request.param]

    def test_has_required_keys(self, flyway):
        assert "label" in flyway
        assert "corridors" in flyway
        assert "species" in flyway
        assert "monthly_calendar" in flyway

    def test_has_corridors(self, flyway):
        assert len(flyway["corridors"]) >= 1
        for c in flyway["corridors"]:
            assert "name" in c
            assert "angle_deg" in c
            assert "sigma" in c
            assert "species" in c

    def test_has_species(self, flyway):
        assert "local" in flyway["species"]
        assert len(flyway["species"]) >= 2

    def test_calendar_has_12_months(self, flyway):
        cal = flyway["monthly_calendar"]
        assert len(cal) == 12
        months = [e["month"] for e in cal]
        assert months == ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def test_calendar_seasons_valid(self, flyway):
        valid_seasons = {"winter", "spring", "summer", "fall"}
        for entry in flyway["monthly_calendar"]:
            assert entry["season"] in valid_seasons

    def test_migration_index_range(self, flyway):
        for entry in flyway["monthly_calendar"]:
            assert 0.0 <= entry["migration_index"] <= 1.0
            assert 0.0 <= entry["intensity"] <= 1.0


class TestSeasonDefaults:
    def test_four_seasons(self):
        assert set(SEASON_DEFAULTS.keys()) == {"winter", "spring", "summer", "fall"}

    def test_all_months_covered(self):
        all_months = set()
        for season in SEASON_DEFAULTS.values():
            all_months.update(season["months"])
        assert all_months == set(range(1, 13))
