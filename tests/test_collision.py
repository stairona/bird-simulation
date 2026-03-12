"""Tests for the multi-factor collision probability model."""

import pytest

from src.core.config import CollisionParams, SeasonDef
from src.core.collision import per_step_collision_prob


@pytest.fixture
def spring_season():
    return SeasonDef(
        name="spring",
        months=[3, 4, 5],
        migration_intensity=1.0,
        resident_fraction=0.45,
        night_fraction=0.60,
        weather_risk=1.20,
    )


@pytest.fixture
def default_params():
    return CollisionParams()


class TestPerStepCollisionProb:
    def test_zero_when_outside_risk_zone(self, spring_season, default_params):
        p = per_step_collision_prob(spring_season, is_night=False, inside_risk_zone=False, params=default_params)
        assert p == 0.0

    def test_positive_when_inside(self, spring_season, default_params):
        p = per_step_collision_prob(spring_season, is_night=False, inside_risk_zone=True, params=default_params)
        assert p > 0.0

    def test_night_increases_risk(self, spring_season, default_params):
        p_day = per_step_collision_prob(spring_season, is_night=False, inside_risk_zone=True, params=default_params)
        p_night = per_step_collision_prob(spring_season, is_night=True, inside_risk_zone=True, params=default_params)
        assert p_night > p_day

    def test_higher_avoidance_less_risk(self, spring_season):
        low_avoid = CollisionParams(avoidance=0.50)
        high_avoid = CollisionParams(avoidance=0.95)
        p_low = per_step_collision_prob(spring_season, is_night=False, inside_risk_zone=True, params=low_avoid)
        p_high = per_step_collision_prob(spring_season, is_night=False, inside_risk_zone=True, params=high_avoid)
        assert p_low > p_high

    def test_capped_at_025(self, spring_season):
        extreme = CollisionParams(base_strike_prob=1.0, avoidance=0.0, night_risk_mult=5.0, altitude_match_prob=1.0)
        p = per_step_collision_prob(spring_season, is_night=True, inside_risk_zone=True, params=extreme)
        assert p <= 0.25
