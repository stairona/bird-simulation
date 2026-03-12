"""Tests for calendar utilities."""

import numpy as np
import pytest

from src.core.calendar import build_month_to_season, migration_index_array


class TestBuildMonthToSeason:
    def test_all_12_months_mapped(self, isabella_cfg):
        mapping = build_month_to_season(isabella_cfg)
        for m in range(1, 13):
            assert m in mapping

    def test_winter_months_correct(self, isabella_cfg):
        mapping = build_month_to_season(isabella_cfg)
        assert mapping[12].name == "winter"
        assert mapping[1].name == "winter"
        assert mapping[2].name == "winter"

    def test_spring_months_correct(self, isabella_cfg):
        mapping = build_month_to_season(isabella_cfg)
        for m in [3, 4, 5]:
            assert mapping[m].name == "spring"


class TestMigrationIndexArray:
    def test_length_12(self, isabella_cfg):
        arr = migration_index_array(isabella_cfg)
        assert len(arr) == 12

    def test_values_in_range(self, isabella_cfg):
        arr = migration_index_array(isabella_cfg)
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

    def test_peak_in_may_or_october(self, isabella_cfg):
        arr = migration_index_array(isabella_cfg)
        peak_month = int(np.argmax(arr))
        # Isabella has peak at Oct (index 9) with 0.85
        assert peak_month in [4, 9]  # May or Oct
