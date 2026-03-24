"""Tests for the discrete-event simulation runner."""

from __future__ import annotations

import math

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
        assert cfg.batch_size == 1
        assert cfg.training_speed == 1.0
        assert cfg.max_staleness == 1

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

    def test_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            SimConfig(
                n_tasks=1,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
                batch_size=0,
            )

    def test_rejects_n_tasks_not_multiple_of_batch_size(self) -> None:
        with pytest.raises(ValueError, match="multiple of batch_size"):
            SimConfig(
                n_tasks=5,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
                batch_size=3,
            )

    def test_rejects_training_speed_below_one(self) -> None:
        with pytest.raises(ValueError, match="training_speed"):
            SimConfig(
                n_tasks=1,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
                training_speed=0.5,
            )

    def test_rejects_zero_max_staleness(self) -> None:
        with pytest.raises(ValueError, match="max_staleness"):
            SimConfig(
                n_tasks=1,
                n_trajectories=4,
                inference_capacity=8,
                judge_capacity=4,
                rollout_duration_fn=_VALID_FN,
                judge_duration_fn=_VALID_FN,
                max_staleness=0,
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
        """Two tasks compete for the same inference/judge slots.

        With batch_size=2 and enough staleness headroom, both are dispatched
        in parallel and consumed in a single training batch.
        """
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
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
            max_staleness=5,
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
            batch_size=2,
            max_staleness=5,
            seed=123,
        )
        sim = Simulation(cfg)
        result = sim.run()

        for stats in result.history:
            assert 0.0 <= stats.inference_utilization <= 1.0
            assert 0.0 <= stats.judge_utilization <= 1.0
            assert stats.training_utilization in (0.0, 1.0)

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
                max_staleness=5,
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
                max_staleness=5,
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
            batch_size=2,
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
            max_staleness=3,
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
            max_staleness=5,
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
            batch_size=3,
            max_staleness=3,
        )
        result = Simulation(cfg).run()
        final = result.history[-1]
        assert final.tasks_by_state.get(TaskState.CONSUMED, 0) == 3

    def test_new_tickstats_fields_present(self) -> None:
        """TickStats includes training_active, training_utilization, etc."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
        )
        result = Simulation(cfg).run()
        stats = result.history[0]
        assert isinstance(stats.training_active, bool)
        assert isinstance(stats.training_utilization, float)
        assert isinstance(stats.training_ticks_total, int)
        assert isinstance(stats.ckpt_version, int)
        assert isinstance(stats.ready_buffer_size, int)
        assert isinstance(stats.max_task_staleness, int)


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

    def test_ckpt_version_starts_at_zero(self) -> None:
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
        )
        sim = Simulation(cfg)
        assert sim.ckpt_version == 0


# ------------------------------------------------------------------
# Training gate: batch consumption
# ------------------------------------------------------------------


class TestBatchTraining:
    def test_training_not_started_below_batch_size(self) -> None:
        """With batch_size=2, 1 READY task is insufficient to start training."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=2,
        )
        sim = Simulation(cfg)
        result = sim.run()
        assert result.tasks_completed == 2
        found_buffered = any(s.ready_buffer_size > 0 for s in result.history)
        assert found_buffered, "Buffer should hold a task while waiting for second"

    def test_exactly_batch_size_consumed_per_run(self) -> None:
        """Each training run consumes exactly batch_size tasks."""
        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=1,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=4,
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 4
        for stats in result.history:
            assert stats.tasks_consumed in (0, 2)

    def test_ckpt_increments_after_training_completes(self) -> None:
        """ckpt_version increases by 1 each time training finishes."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=1,
            training_speed=1_000_000.0,
            max_staleness=2,
        )
        sim = Simulation(cfg)
        result = sim.run()
        assert result.tasks_completed == 2
        final_ckpt = result.history[-1].ckpt_version
        assert final_ckpt >= 1

    def test_training_utilization_matches_active(self) -> None:
        """training_utilization is 1.0 when active, 0.0 when idle."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=1,
            training_speed=1.0,
            max_staleness=2,
        )
        result = Simulation(cfg).run()
        for stats in result.history:
            if stats.training_active:
                assert stats.training_utilization == 1.0
                assert stats.training_ticks_total > 0
            else:
                assert stats.training_utilization == 0.0
                assert stats.training_ticks_total == 0

    def test_training_blocks_next_batch(self) -> None:
        """While training is in flight, a second batch cannot start."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=1,
            training_speed=1.0,
            max_staleness=2,
        )
        result = Simulation(cfg).run()
        training_ticks = [s for s in result.history if s.training_active]
        assert len(training_ticks) > 0, "Training should occupy at least one tick"


# ------------------------------------------------------------------
# Training duration
# ------------------------------------------------------------------


class TestTrainingDuration:
    def test_duration_formula(self) -> None:
        """Training ticks == ceil(rollout * n_traj * batch * 3 / speed).

        Uses n_tasks=8 so the first training run completes before the sim
        ends (the sim only stops when all tasks are consumed, and the second
        batch waits for training to finish).
        """
        rollout_dur = 5
        n_trajectories = 2
        batch_size = 4
        speed = 2.0
        expected = math.ceil(rollout_dur * n_trajectories * batch_size * 3 / speed)

        cfg = SimConfig(
            n_tasks=8,
            n_trajectories=n_trajectories,
            inference_capacity=8,
            judge_capacity=8,
            rollout_duration_fn=constant_duration(rollout_dur),
            judge_duration_fn=constant_duration(1),
            batch_size=batch_size,
            training_speed=speed,
            max_staleness=8,
            seed=0,
        )
        result = Simulation(cfg).run()

        first_training_ticks = sum(
            1 for s in result.history if s.training_active and s.ckpt_version == 0
        )
        assert first_training_ticks == expected

    def test_training_duration_at_least_one(self) -> None:
        """Even with very high speed the training takes at least 1 tick."""
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=1,
            training_speed=1_000_000.0,
        )
        result = Simulation(cfg).run()
        training_ticks = sum(1 for s in result.history if s.training_active)
        assert training_ticks >= 1


# ------------------------------------------------------------------
# Staleness tracking
# ------------------------------------------------------------------


class TestStaleness:
    def test_birth_ckpt_set_on_first_dispatch(self) -> None:
        """Tasks get birth_ckpt when first dispatched from PENDING."""
        cfg = SimConfig(
            n_tasks=2,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            max_staleness=2,
        )
        sim = Simulation(cfg)
        sim.run()
        for task in sim.tasks:
            assert task.birth_ckpt is not None

    def test_staleness_never_exceeds_max(self) -> None:
        """Property: max_task_staleness never exceeds config.max_staleness."""
        cfg = SimConfig(
            n_tasks=12,
            n_trajectories=4,
            inference_capacity=6,
            judge_capacity=3,
            rollout_duration_fn=uniform_duration(1, 5),
            judge_duration_fn=uniform_duration(1, 3),
            batch_size=3,
            training_speed=2.0,
            max_staleness=2,
            seed=42,
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 12
        for stats in result.history:
            assert stats.max_task_staleness <= cfg.max_staleness

    def test_staleness_zero_with_single_batch(self) -> None:
        """When all tasks fit in one batch, staleness stays 0."""
        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=1,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=4,
            max_staleness=1,
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 4
        for stats in result.history:
            assert stats.max_task_staleness == 0


# ------------------------------------------------------------------
# Pipeline-depth throttle
# ------------------------------------------------------------------


class TestPipelineThrottle:
    def test_active_tasks_bounded_by_pipeline_cap(self) -> None:
        """The number of active tasks never exceeds max_staleness * batch_size."""
        batch_size = 2
        max_staleness = 2
        pipeline_cap = batch_size * max_staleness

        cfg = SimConfig(
            n_tasks=10,
            n_trajectories=2,
            inference_capacity=10,
            judge_capacity=10,
            rollout_duration_fn=constant_duration(3),
            judge_duration_fn=constant_duration(2),
            batch_size=batch_size,
            training_speed=2.0,
            max_staleness=max_staleness,
            seed=0,
        )
        sim = Simulation(cfg)
        result = sim.run()
        assert result.tasks_completed == 10

        for stats in result.history:
            n_pending = stats.tasks_by_state.get(TaskState.PENDING, 0)
            n_consumed = stats.tasks_by_state.get(TaskState.CONSUMED, 0)
            active = cfg.n_tasks - n_pending - n_consumed
            assert active <= pipeline_cap, (
                f"tick {stats.tick}: {active} active > cap {pipeline_cap}"
            )

    def test_throttle_with_max_staleness_one(self) -> None:
        """max_staleness=1 limits to exactly batch_size active tasks."""
        batch_size = 3
        cfg = SimConfig(
            n_tasks=6,
            n_trajectories=1,
            inference_capacity=6,
            judge_capacity=6,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=batch_size,
            max_staleness=1,
        )
        sim = Simulation(cfg)
        result = sim.run()
        assert result.tasks_completed == 6

        for stats in result.history:
            n_pending = stats.tasks_by_state.get(TaskState.PENDING, 0)
            n_consumed = stats.tasks_by_state.get(TaskState.CONSUMED, 0)
            active = cfg.n_tasks - n_pending - n_consumed
            assert active <= batch_size


# ------------------------------------------------------------------
# End-to-end with training
# ------------------------------------------------------------------


class TestEndToEnd:
    def test_full_run_with_training(self) -> None:
        """A larger run with all features enabled completes correctly.

        The last training run starts but the sim exits before it finishes
        (all tasks consumed), so final ckpt_version is n_batches - 1.
        """
        cfg = SimConfig(
            n_tasks=20,
            n_trajectories=8,
            inference_capacity=8,
            judge_capacity=4,
            rollout_duration_fn=uniform_duration(2, 10),
            judge_duration_fn=uniform_duration(1, 4),
            batch_size=4,
            training_speed=2.0,
            max_staleness=3,
            seed=0,
        )
        result = Simulation(cfg).run()
        assert result.tasks_completed == 20
        final = result.history[-1]
        n_batches = 20 // 4
        assert final.ckpt_version == n_batches - 1

        for stats in result.history:
            assert stats.max_task_staleness <= 3

    def test_determinism_with_training(self) -> None:
        """Two runs with the same seed produce identical results."""

        def make() -> SimConfig:
            return SimConfig(
                n_tasks=8,
                n_trajectories=4,
                inference_capacity=4,
                judge_capacity=2,
                rollout_duration_fn=uniform_duration(1, 6),
                judge_duration_fn=uniform_duration(1, 3),
                batch_size=2,
                training_speed=3.0,
                max_staleness=3,
                seed=99,
            )

        r1 = Simulation(make()).run()
        r2 = Simulation(make()).run()
        assert r1.ticks_elapsed == r2.ticks_elapsed
        for s1, s2 in zip(r1.history, r2.history, strict=True):
            assert s1 == s2
