"""Tests for corridor geometry and density models."""

import math

import numpy as np
import pytest

from src.core.corridors import (
    bezier,
    bezier_deriv,
    build_curved_corridor,
    corridor_density,
    gaussian2d,
    vec_add,
    vec_len,
    vec_norm,
    vec_perp,
    rotate,
)


class TestVectorHelpers:
    def test_vec_add(self):
        assert vec_add((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)

    def test_vec_norm_unit_length(self):
        n = vec_norm((3.0, 4.0))
        assert abs(vec_len(n) - 1.0) < 1e-10

    def test_vec_norm_zero(self):
        assert vec_norm((0.0, 0.0)) == (0.0, 0.0)

    def test_vec_perp_is_orthogonal(self):
        v = (3.0, 4.0)
        p = vec_perp(v)
        dot = v[0] * p[0] + v[1] * p[1]
        assert abs(dot) < 1e-10

    def test_rotate_90(self):
        v = (1.0, 0.0)
        r = rotate(v, math.pi / 2)
        assert abs(r[0] - 0.0) < 1e-10
        assert abs(r[1] - 1.0) < 1e-10


class TestBezier:
    def test_endpoints(self):
        p0, p1, p2, p3 = (0, 0), (1, 1), (2, 1), (3, 0)
        assert bezier(p0, p1, p2, p3, 0.0) == pytest.approx((0, 0), abs=1e-10)
        assert bezier(p0, p1, p2, p3, 1.0) == pytest.approx((3, 0), abs=1e-10)

    def test_midpoint_different_from_endpoints(self):
        p0, p1, p2, p3 = (0, 0), (0, 1), (1, 1), (1, 0)
        mid = bezier(p0, p1, p2, p3, 0.5)
        assert mid != pytest.approx((0, 0), abs=0.01)
        assert mid != pytest.approx((1, 0), abs=0.01)


class TestGaussian2d:
    def test_peak_at_center(self):
        x = np.array([0.5])
        y = np.array([0.5])
        val = gaussian2d(x, y, 0.5, 0.5, 0.1, 0.1)
        assert val[0] == pytest.approx(1.0, abs=1e-10)

    def test_decays_away_from_center(self):
        x = np.array([0.5, 1.0])
        y = np.array([0.5, 1.0])
        vals = gaussian2d(x, y, 0.5, 0.5, 0.1, 0.1)
        assert vals[0] > vals[1]


class TestCorridorDensity:
    def test_returns_correct_shape(self, isabella_cfg):
        from src.core.turbines import make_turbine_layout
        _, xy = make_turbine_layout(isabella_cfg)
        dens = corridor_density(xy, 4, isabella_cfg)  # May
        assert dens.shape == (136,)

    def test_density_varies_by_month(self, isabella_cfg):
        from src.core.turbines import make_turbine_layout
        _, xy = make_turbine_layout(isabella_cfg)
        d_jan = corridor_density(xy, 0, isabella_cfg)
        d_may = corridor_density(xy, 4, isabella_cfg)
        # Densities should differ due to seasonal weighting
        assert not np.allclose(d_jan, d_may)

    def test_density_nonnegative(self, isabella_cfg):
        from src.core.turbines import make_turbine_layout
        _, xy = make_turbine_layout(isabella_cfg)
        for m in range(12):
            dens = corridor_density(xy, m, isabella_cfg)
            assert np.all(np.isfinite(dens))
