"""Tests for turbine layout and avoidance models."""

import numpy as np
import pytest

from src.core.turbines import make_turbine_layout, turbine_avoidance_factor


class TestMakeTurbineLayout:
    def test_correct_count(self, isabella_cfg):
        ids, xy = make_turbine_layout(isabella_cfg)
        assert len(ids) == 136
        assert xy.shape == (136, 2)

    def test_ids_are_sequential(self, isabella_cfg):
        ids, _ = make_turbine_layout(isabella_cfg)
        np.testing.assert_array_equal(ids, np.arange(1, 137))

    def test_positions_in_unit_square(self, isabella_cfg):
        _, xy = make_turbine_layout(isabella_cfg)
        assert xy.min() >= 0.0
        assert xy.max() <= 1.0

    def test_deterministic_with_seed(self, isabella_cfg):
        _, xy1 = make_turbine_layout(isabella_cfg)
        _, xy2 = make_turbine_layout(isabella_cfg)
        np.testing.assert_array_equal(xy1, xy2)


class TestTurbineAvoidanceFactor:
    def test_normal_output_range(self, isabella_cfg):
        _, xy = make_turbine_layout(isabella_cfg)
        avoid = turbine_avoidance_factor(xy)
        assert len(avoid) == 136
        assert avoid.min() >= 0.0
        assert avoid.max() <= 1.0

    def test_single_turbine_no_crash(self):
        xy = np.array([[0.5, 0.5]])
        avoid = turbine_avoidance_factor(xy)
        assert len(avoid) == 1
        assert not np.isnan(avoid[0])
        assert avoid[0] == pytest.approx(0.30)

    def test_empty_array(self):
        xy = np.empty((0, 2))
        avoid = turbine_avoidance_factor(xy)
        assert len(avoid) == 0

    def test_two_turbines(self):
        xy = np.array([[0.2, 0.3], [0.8, 0.7]])
        avoid = turbine_avoidance_factor(xy)
        assert len(avoid) == 2
        assert all(not np.isnan(a) for a in avoid)
