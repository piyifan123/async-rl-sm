"""Tests for the discrete-event simulation runner."""

from __future__ import annotations

import numpy as np
import pytest

from async_gym.simulation import (
    SimConfig,
    Simulation,
    constant_duration,
    uniform_duration,
)
from async_gym.task import TaskState

# ------------------------------------------------------------------
# Duration helper tests
# ------------------------------------------------------------------


class TestConstantDuration:
    def test_produces_correct_values(self) -> None:
        fn = constant_duration(5)
        result = fn(3, np.random.default_rng(0))
        assert result == [5, 5, 5]

    def test_single_item(self) -> None:
        fn = constant_duration(1)
        assert fn(1, np.random.default_rng(0)) == [1]

    def test_zero_count(self) -> None:
        fn = constant_duration(3)
        assert fn(0, np.random.default_rng(0)) == []

    def test_rejects_zero_value(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            constant_duration(0)

    def test_rejects_negative_value(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            constant_duration(-1)


class TestUniformDuration:
    def test_values_within_range(self) -> None:
        fn = uniform_duration(2, 10)
        rng = np.random.default_rng(42)
        values = fn(100, rng)
        assert all(2 <= v <= 10 for v in values)
        assert len(values) == 100

    def test_single_value_range(self) -> None:
        fn = uniform_duration(3, 3)
        values = fn(5, np.random.default_rng(0))
        assert values == [3, 3, 3, 3, 3]

    def test_returns_ints(self) -> None:
        fn = uniform_duration(1, 5)
        values = fn(10, np.random.default_rng(0))
        assert all(isinstance(v, int) for v in values)

    def test_rejects_low_below_one(self) -> None:
        with pytest.raises(ValueError, match="low must be >= 1"):
            uniform_duration(0, 5)

    def test_rejects_high_below_low(self) -> None:
        with pytest.raises(ValueError, match="high must be >= low"):
            uniform_duration(5, 3)


# ------------------------------------------------------------------
# SimConfig validation
# ------------------------------------------------------------------

_VALID_FN = constant_duration(1)


class TestSimConfig:
    def test_valid_config(self) -> None:
        cfg = SimConfig(
            n_tasks=5,
            n_trajectories=4,
            inference_capacity=8,
            judge_capacity=4,
            rollout_duration_fn=_VALID_FN,
            judge_duration_fn=_VALID_FN,
        )
        assert cfg.n_tasks == 5
        assert cfg.max_ticks == 100_000
        assert cfg.seed is None

    def test_rejects_zero_n_tasks(self) -> None:
        with pytest.raises(ValueError, match="n_tasks"):
            SimConfig(
                n_tasks=0,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
            )

    def test_rejects_zero_n_trajectories(self) -> None:
        with pytest.raises(ValueError, match="n_trajectories"):
            SimConfig(
                n_tasks=1,
                n_trajectories=0,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
            )

    def test_rejects_zero_inference_capacity(self) -> None:
        with pytest.raises(ValueError, match="inference_capacity"):
            SimConfig(
                n_tasks=1,
                n_trajectories=4,
                inference_capacity=0,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
            )

    def test_rejects_zero_judge_capacity(self) -> None:
        with pytest.raises(ValueError, match="judge_capacity"):
            SimConfig(
                n_tasks=1,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=0,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
            )

    def test_rejects_zero_max_ticks(self) -> None:
        with pytest.raises(ValueError, match="max_ticks"):
            SimConfig(
                n_tasks=1,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
                max_ticks=0,
            )


# ------------------------------------------------------------------
# Trivial simulation (1 task, 1 trajectory, constant durations)
# ------------------------------------------------------------------


class TestTrivialSimulation:
    def test_single_task_single_trajectory(self) -> None:
        """1 task, 1 trajectory, rollout=1 tick, judge=1 tick.

        Expected timeline:
          tick 0: dispatch rollout, tick it -> completes, dispatch judge, NO:
                  dispatch happens first, then tick.
          tick 0: dispatch rollout (1 tick), tick -> completes, judge pending
          tick 1: dispatch judge (1 tick), tick -> completes, consume
          Total: 2 ticks.
        """
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 1
        assert result.ticks_elapsed == 2

    def test_single_task_longer_durations(self) -> None:
        """1 task, 1 trajectory, rollout=3 ticks, judge=2 ticks -> 5 ticks total."""
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(3),
            judge_duration_fn=constant_duration(2),
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 1
        assert result.ticks_elapsed == 5

    def test_two_trajectories_sufficient_capacity(self) -> None:
        """1 task, 2 trajectories, capacity=2: both dispatch in parallel.

        rollout=1, judge=1 -> 2 ticks total.
        """
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=2,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 1
        assert result.ticks_elapsed == 2


# ------------------------------------------------------------------
# Multi-task simulation
# ------------------------------------------------------------------


class TestMultiTaskSimulation:
    def test_two_tasks_share_capacity(self) -> None:
        """Two tasks compete for the same inference/judge slots."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 2
        assert result.ticks_elapsed == 2

    def test_limited_capacity_serialises_work(self) -> None:
        """With capacity=1 and 2 tasks x 1 trajectory, work is serialised."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 2
        assert result.ticks_elapsed >= 2

    def test_all_tasks_reach_consumed(self) -> None:
        cfg = SimConfig(
            n_tasks=5,
            n_trajectories=3,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(2),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 5


# ------------------------------------------------------------------
# Pool capacity invariant
# ------------------------------------------------------------------


class TestPoolInvariant:
    def test_pools_never_over_acquired(self) -> None:
        """Run a simulation and check that pools are consistent after each tick."""
        cfg = SimConfig(
            n_tasks=10,
            n_trajectories=4,
            inference_capacity=6,
            judge_capacity=3,
            rollout_duration_fn=uniform_duration(1, 5),
            judge_duration_fn=uniform_duration(1, 3),
            seed=123,
        )
        sim = Simulation(cfg)
        result = sim.run()

        for stats in result.history:
            assert 0.0 <= stats.inference_utilization <= 1.0
            assert 0.0 <= stats.judge_utilization <= 1.0

        assert sim.inference_pool.in_use == 0
        assert sim.judge_pool.in_use == 0
        assert result.tasks_completed == 10


# ------------------------------------------------------------------
# Determinism / seed
# ------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        def make_config(seed: int) -> SimConfig:
            return SimConfig(
                n_tasks=5,
                n_trajectories=4,
                inference_capacity=4,
                judge_capacity=2,
                rollout_duration_fn=uniform_duration(1, 8),
                judge_duration_fn=uniform_duration(1, 4),
                seed=seed,
            )

        r1 = Simulation(make_config(42)).run()
        r2 = Simulation(make_config(42)).run()

        assert r1.ticks_elapsed == r2.ticks_elapsed
        assert r1.tasks_completed == r2.tasks_completed
        for s1, s2 in zip(r1.history, r2.history, strict=True):
            assert s1 == s2

    def test_different_seed_may_differ(self) -> None:
        def make_config(seed: int) -> SimConfig:
            return SimConfig(
                n_tasks=5,
                n_trajectories=8,
                inference_capacity=4,
                judge_capacity=2,
                rollout_duration_fn=uniform_duration(1, 10),
                judge_duration_fn=uniform_duration(1, 5),
                seed=seed,
            )

        r1 = Simulation(make_config(1)).run()
        r2 = Simulation(make_config(999)).run()
        assert r1.tasks_completed == r2.tasks_completed == 5
        histories_differ = any(
            s1.rollouts_dispatched != s2.rollouts_dispatched
            for s1, s2 in zip(r1.history, r2.history, strict=False)
        )
        assert histories_differ


# ------------------------------------------------------------------
# Max ticks safety
# ------------------------------------------------------------------


class TestMaxTicks:
    def test_stops_at_max_ticks(self) -> None:
        cfg = SimConfig(
            n_tasks=100,
            n_trajectories=100,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(10),
            judge_duration_fn=constant_duration(10),
            max_ticks=5,
        )
        result = Simulation(cfg).run()
        assert result.ticks_elapsed == 5
        assert result.tasks_completed < 100

    def test_partial_progress_recorded(self) -> None:
        cfg = SimConfig(
            n_tasks=10,
            n_trajectories=4,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            max_ticks=3,
        )
        result = Simulation(cfg).run()
        assert result.ticks_elapsed == 3
        assert len(result.history) == 3


# ------------------------------------------------------------------
# TickStats content
# ------------------------------------------------------------------


class TestTickStats:
    def test_history_length_matches_ticks(self) -> None:
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        assert len(result.history) == result.ticks_elapsed

    def test_tick_indices_sequential(self) -> None:
        cfg = SimConfig(
            n_tasks=3,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        for i, stats in enumerate(result.history):
            assert stats.tick == i

    def test_tasks_by_state_sums_to_n_tasks(self) -> None:
        cfg = SimConfig(
            n_tasks=5,
            n_trajectories=3,
            inference_capacity=4,
            judge_capacity=2,
            rollout_duration_fn=uniform_duration(1, 4),
            judge_duration_fn=uniform_duration(1, 3),
            seed=7,
        )
        result = Simulation(cfg).run()
        for stats in result.history:
            assert sum(stats.tasks_by_state.values()) == 5

    def test_final_tick_all_consumed(self) -> None:
        cfg = SimConfig(
            n_tasks=3,
            n_trajectories=2,
            inference_capacity=6,
            judge_capacity=6,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        result = Simulation(cfg).run()
        final = result.history[-1]
        assert final.tasks_by_state.get(TaskState.CONSUMED, 0) == 3


# ------------------------------------------------------------------
# Simulation object properties
# ------------------------------------------------------------------


class TestSimulationProperties:
    def test_tasks_created_with_correct_ids(self) -> None:
        cfg = SimConfig(
            n_tasks=3,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        sim = Simulation(cfg)
        assert [t.task_id for t in sim.tasks] == ["task-0", "task-1", "task-2"]

    def test_pools_have_correct_capacity(self) -> None:
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=8,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        sim = Simulation(cfg)
        assert sim.inference_pool.capacity == 8
        assert sim.judge_pool.capacity == 4

    def test_config_accessible(self) -> None:
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        sim = Simulation(cfg)
        assert sim.config is cfg
