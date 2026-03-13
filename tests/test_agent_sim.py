"""Tests for the agent-based collision simulation."""

import numpy as np
import pytest

from src.phase2_mortality.agent_sim import (
    simulate_agent,
    _spawn_migrants,
    _spawn_residents,
)
from src.core.corridors import corridors_to_world_space


class TestSimulateAgent:
    def test_returns_four_arrays(self, isabella_cfg):
        daily_birds, daily_deaths, heat, turbines = simulate_agent(isabella_cfg)
        assert daily_birds.shape == (365,)
        assert daily_deaths.shape == (365,)
        assert heat.ndim == 2
        assert turbines.ndim == 2 and turbines.shape[1] == 2

    def test_nonnegative(self, isabella_cfg):
        daily_birds, daily_deaths, heat, turbines = simulate_agent(isabella_cfg)
        assert (daily_birds >= 0).all()
        assert (daily_deaths >= 0).all()
        assert (heat >= 0).all()

    def test_deaths_le_birds(self, isabella_cfg):
        daily_birds, daily_deaths, _, _ = simulate_agent(isabella_cfg)
        assert (daily_deaths <= daily_birds).all()

    def test_deterministic(self, isabella_cfg):
        r1 = simulate_agent(isabella_cfg)
        r2 = simulate_agent(isabella_cfg)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_turbine_count_matches_config(self, isabella_cfg):
        _, _, _, turbines = simulate_agent(isabella_cfg)
        assert len(turbines) == isabella_cfg.turbine_count


class TestCorridorsToWorldSpace:
    def test_returns_list_of_dicts(self, isabella_cfg):
        result = corridors_to_world_space(isabella_cfg)
        assert isinstance(result, list)
        assert len(result) == len(isabella_cfg.corridors)
        for c in result:
            assert "name" in c
            assert "p0" in c
            assert "p1" in c
            assert "sigma" in c

    def test_within_world_bounds(self, isabella_cfg):
        W, H = isabella_cfg.simulation.agent.world_size
        for c in corridors_to_world_space(isabella_cfg):
            assert 0 <= c["p0"][0] <= W
            assert 0 <= c["p0"][1] <= H
            assert 0 <= c["p1"][0] <= W
            assert 0 <= c["p1"][1] <= H


class TestSpawnMigrants:
    def test_correct_count(self, isabella_cfg):
        corridors = corridors_to_world_space(isabella_cfg)
        W, H = isabella_cfg.simulation.agent.world_size
        rng = np.random.default_rng(42)
        birds = _spawn_migrants(20, corridors, 2.4, W, H, rng)
        assert len(birds) == 20

    def test_all_alive(self, isabella_cfg):
        corridors = corridors_to_world_space(isabella_cfg)
        W, H = isabella_cfg.simulation.agent.world_size
        rng = np.random.default_rng(42)
        birds = _spawn_migrants(10, corridors, 2.4, W, H, rng)
        assert all(b["alive"] for b in birds)
        assert all(b["type"] == "migrant" for b in birds)


class TestSpawnResidents:
    def test_correct_count(self):
        rng = np.random.default_rng(42)
        birds = _spawn_residents(15, 1.2, 100, 100, rng)
        assert len(birds) == 15

    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        birds = _spawn_residents(30, 1.2, 100, 100, rng)
        for b in birds:
            assert 0 <= b["pos"][0] <= 100
            assert 0 <= b["pos"][1] <= 100
            assert b["type"] == "resident"
