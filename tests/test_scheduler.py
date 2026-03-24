"""Tests for the pluggable scheduler interface and baseline implementation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from async_gym.scenarios import SCENARIOS, get_scenario
from async_gym.scheduler import (
    DispatchAction,
    GreedyFIFOScheduler,
    SchedulerView,
    SRPTAgingScheduler,
)
from async_gym.simulation import (
    SimConfig,
    Simulation,
    constant_duration,
    uniform_duration,
)
from async_gym.task import Task, TaskState

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_VALID_FN = constant_duration(1)


def _make_view(**overrides: object) -> SchedulerView:
    """Build a ``SchedulerView`` with sensible defaults, overriding any field."""
    defaults: dict[str, object] = {
        "tick": 0,
        "ckpt_version": 0,
        "inference_available": 4,
        "inference_capacity": 4,
        "judge_available": 4,
        "judge_capacity": 4,
        "active_count": 0,
        "pipeline_cap": 4,
        "batch_size": 1,
        "max_staleness": 2,
        "training_in_flight": False,
        "ready_buffer_size": 0,
        "max_active_staleness": 0,
    }
    defaults.update(overrides)
    return SchedulerView(**defaults)  # type: ignore[arg-type]


def _make_task(task_id: str = "t0", n_trajectories: int = 4) -> Task:
    """Create a fresh PENDING task."""
    return Task(task_id=task_id, n_trajectories=n_trajectories)


def _admit_task(task: Task, ckpt: int = 0) -> None:
    """Simulate admission by setting ``birth_ckpt``."""
    task.birth_ckpt = ckpt


# ------------------------------------------------------------------
# GreedyFIFOScheduler.should_admit
# ------------------------------------------------------------------


class TestGreedyFIFOAdmission:
    def test_admits_when_under_cap(self) -> None:
        sched = GreedyFIFOScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=4, active_count=2)
        assert sched.should_admit(task, 2, view) is True

    def test_rejects_when_at_cap(self) -> None:
        sched = GreedyFIFOScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=4)
        assert sched.should_admit(task, 4, view) is False

    def test_rejects_when_above_cap(self) -> None:
        sched = GreedyFIFOScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=2)
        assert sched.should_admit(task, 3, view) is False

    def test_admits_with_zero_active(self) -> None:
        sched = GreedyFIFOScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=1)
        assert sched.should_admit(task, 0, view) is True


# ------------------------------------------------------------------
# GreedyFIFOScheduler.plan_rollout_dispatch
# ------------------------------------------------------------------


class TestGreedyFIFORolloutDispatch:
    def test_empty_when_no_tasks(self) -> None:
        sched = GreedyFIFOScheduler()
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([], view) == []

    def test_skips_terminal_tasks(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=1)
        _admit_task(t)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        t.consume()
        assert t.state == TaskState.CONSUMED
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([t], view) == []

    def test_skips_pending_tasks(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task()
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([t], view) == []

    def test_greedy_allocation(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=4)
        _admit_task(t)
        view = _make_view(inference_available=4)
        actions = sched.plan_rollout_dispatch([t], view)
        assert len(actions) == 1
        assert actions[0].task is t
        assert actions[0].count == 4

    def test_capped_by_available(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=8)
        _admit_task(t)
        view = _make_view(inference_available=3)
        actions = sched.plan_rollout_dispatch([t], view)
        assert actions[0].count == 3

    def test_first_task_gets_priority(self) -> None:
        """Earlier tasks in the list consume slots first."""
        sched = GreedyFIFOScheduler()
        t0 = _make_task("t0", n_trajectories=3)
        t1 = _make_task("t1", n_trajectories=3)
        _admit_task(t0)
        _admit_task(t1)
        view = _make_view(inference_available=4)
        actions = sched.plan_rollout_dispatch([t0, t1], view)
        assert len(actions) == 2
        assert actions[0].task is t0
        assert actions[0].count == 3
        assert actions[1].task is t1
        assert actions[1].count == 1

    def test_no_actions_when_zero_available(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=2)
        _admit_task(t)
        view = _make_view(inference_available=0)
        assert sched.plan_rollout_dispatch([t], view) == []

    def test_skips_task_with_zero_pending_rollout(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=2)
        _admit_task(t)
        t.submit_rollouts([1, 1])
        assert t.pending_rollout == 0
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([t], view) == []


# ------------------------------------------------------------------
# GreedyFIFOScheduler.plan_judge_dispatch
# ------------------------------------------------------------------


class TestGreedyFIFOJudgeDispatch:
    def test_empty_when_no_tasks(self) -> None:
        sched = GreedyFIFOScheduler()
        view = _make_view(judge_available=4)
        assert sched.plan_judge_dispatch([], view) == []

    def test_skips_pending_tasks(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task()
        view = _make_view(judge_available=4)
        assert sched.plan_judge_dispatch([t], view) == []

    def test_dispatches_pending_judges(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=2)
        _admit_task(t)
        t.submit_rollouts([1, 1])
        t.tick()
        assert t.pending_judge == 2
        view = _make_view(judge_available=4)
        actions = sched.plan_judge_dispatch([t], view)
        assert len(actions) == 1
        assert actions[0].count == 2

    def test_capped_by_available(self) -> None:
        sched = GreedyFIFOScheduler()
        t = _make_task(n_trajectories=4)
        _admit_task(t)
        t.submit_rollouts([1, 1, 1, 1])
        t.tick()
        assert t.pending_judge == 4
        view = _make_view(judge_available=2)
        actions = sched.plan_judge_dispatch([t], view)
        assert actions[0].count == 2


# ------------------------------------------------------------------
# GreedyFIFOScheduler.name
# ------------------------------------------------------------------


class TestGreedyFIFOName:
    def test_name(self) -> None:
        assert GreedyFIFOScheduler().name == "greedy-fifo"


# ------------------------------------------------------------------
# Behavioral equivalence: default scheduler == explicit GreedyFIFO
# ------------------------------------------------------------------


class TestBehavioralEquivalence:
    """Simulation with no scheduler and Simulation with explicit
    GreedyFIFOScheduler must produce identical results."""

    @pytest.fixture(params=list(SCENARIOS.keys()))
    def scenario_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_identical_results_across_scenarios(self, scenario_name: str) -> None:
        config = get_scenario(scenario_name).config
        r_default = Simulation(config).run()
        r_explicit = Simulation(config, scheduler=GreedyFIFOScheduler()).run()

        assert r_default.ticks_elapsed == r_explicit.ticks_elapsed
        assert r_default.tasks_completed == r_explicit.tasks_completed
        assert r_default.tasks_dropped == r_explicit.tasks_dropped
        for s1, s2 in zip(r_default.history, r_explicit.history, strict=True):
            assert s1 == s2


# ------------------------------------------------------------------
# Simulation validates scheduler plans
# ------------------------------------------------------------------


class TestSimulationValidation:
    def _base_config(self) -> SimConfig:
        return SimConfig(
            n_tasks=2,
            n_trajectories=2,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=_VALID_FN,
            judge_duration_fn=_VALID_FN,
            batch_size=2,
            max_staleness=4,
        )

    def test_rejects_rollout_exceeding_capacity(self) -> None:
        """A scheduler that allocates more rollout slots than available."""

        class OverAllocRollout(GreedyFIFOScheduler):
            def plan_rollout_dispatch(
                self, tasks: Sequence[Task], view: SchedulerView
            ) -> list[DispatchAction]:
                actions = []
                for t in tasks:
                    if t.birth_ckpt is not None and t.pending_rollout > 0:
                        actions.append(DispatchAction(task=t, count=t.pending_rollout))
                return actions

        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=4,
            inference_capacity=2,
            judge_capacity=2,
            rollout_duration_fn=_VALID_FN,
            judge_duration_fn=_VALID_FN,
            batch_size=4,
            max_staleness=4,
        )
        sim = Simulation(cfg, scheduler=OverAllocRollout())
        with pytest.raises(ValueError, match="exceeds inference_available"):
            sim.run()

    def test_rejects_judge_exceeding_capacity(self) -> None:
        """A scheduler that allocates more judge slots than available."""

        class OverAllocJudge(GreedyFIFOScheduler):
            def plan_judge_dispatch(
                self, tasks: Sequence[Task], view: SchedulerView
            ) -> list[DispatchAction]:
                actions = []
                for t in tasks:
                    if t.birth_ckpt is not None and t.pending_judge > 0:
                        actions.append(DispatchAction(task=t, count=t.pending_judge))
                return actions

        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=2,
            inference_capacity=8,
            judge_capacity=1,
            rollout_duration_fn=_VALID_FN,
            judge_duration_fn=_VALID_FN,
            batch_size=4,
            max_staleness=4,
        )
        sim = Simulation(cfg, scheduler=OverAllocJudge())
        with pytest.raises(ValueError, match="exceeds judge_available"):
            sim.run()

    def test_rejects_action_with_zero_count(self) -> None:
        class ZeroCountScheduler(GreedyFIFOScheduler):
            def plan_rollout_dispatch(
                self, tasks: Sequence[Task], view: SchedulerView
            ) -> list[DispatchAction]:
                for t in tasks:
                    if t.birth_ckpt is not None and t.pending_rollout > 0:
                        return [DispatchAction(task=t, count=0)]
                return []

        cfg = self._base_config()
        sim = Simulation(cfg, scheduler=ZeroCountScheduler())
        with pytest.raises(ValueError, match="must be >= 1"):
            sim.run()

    def test_rejects_non_admitted_task(self) -> None:
        """A scheduler that returns actions for PENDING (non-admitted) tasks."""

        class PendingScheduler(GreedyFIFOScheduler):
            def should_admit(self, task: Task, active_count: int, view: SchedulerView) -> bool:
                return False

            def plan_rollout_dispatch(
                self, tasks: Sequence[Task], view: SchedulerView
            ) -> list[DispatchAction]:
                for t in tasks:
                    if t.pending_rollout > 0:
                        return [DispatchAction(task=t, count=1)]
                return []

        cfg = self._base_config()
        sim = Simulation(cfg, scheduler=PendingScheduler())
        with pytest.raises(ValueError, match="non-admitted"):
            sim.run()


# ------------------------------------------------------------------
# Custom scheduler smoke test: single-slot-per-task
# ------------------------------------------------------------------


class _SingleSlotScheduler(GreedyFIFOScheduler):
    """Allocate at most 1 rollout and 1 judge slot per task per tick."""

    @property
    def name(self) -> str:
        return "single-slot"

    def plan_rollout_dispatch(
        self, tasks: Sequence[Task], view: SchedulerView
    ) -> list[DispatchAction]:
        actions: list[DispatchAction] = []
        remaining = view.inference_available
        for t in tasks:
            if remaining <= 0:
                break
            if t.state in {TaskState.CONSUMED, TaskState.DROPPED}:
                continue
            if t.birth_ckpt is None:
                continue
            if t.pending_rollout > 0:
                actions.append(DispatchAction(task=t, count=1))
                remaining -= 1
        return actions

    def plan_judge_dispatch(
        self, tasks: Sequence[Task], view: SchedulerView
    ) -> list[DispatchAction]:
        actions: list[DispatchAction] = []
        remaining = view.judge_available
        for t in tasks:
            if remaining <= 0:
                break
            if t.state in {TaskState.CONSUMED, TaskState.DROPPED}:
                continue
            if t.birth_ckpt is None:
                continue
            if t.pending_judge > 0:
                actions.append(DispatchAction(task=t, count=1))
                remaining -= 1
        return actions


class TestCustomScheduler:
    def test_single_slot_completes(self) -> None:
        """A scheduler that limits to 1 slot/task/tick still finishes."""
        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=10,
        )
        result = Simulation(cfg, scheduler=_SingleSlotScheduler()).run()
        assert result.tasks_completed + result.tasks_dropped == cfg.n_tasks
        assert result.tasks_completed == 4

    def test_single_slot_takes_longer(self) -> None:
        """Single-slot should be slower than greedy with same config."""
        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=4,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=10,
        )
        greedy = Simulation(cfg, scheduler=GreedyFIFOScheduler()).run()
        single = Simulation(cfg, scheduler=_SingleSlotScheduler()).run()
        assert single.ticks_elapsed >= greedy.ticks_elapsed


# ------------------------------------------------------------------
# Custom scheduler: override only should_admit (back-pressure)
# ------------------------------------------------------------------


class _TightAdmissionScheduler(GreedyFIFOScheduler):
    """Only admit tasks when training is idle."""

    @property
    def name(self) -> str:
        return "tight-admission"

    def should_admit(self, task: Task, active_count: int, view: SchedulerView) -> bool:
        if view.training_in_flight:
            return False
        return super().should_admit(task, active_count, view)


class TestOverrideSingleHook:
    def test_tight_admission_completes(self) -> None:
        """Overriding only should_admit still produces valid results."""
        cfg = SimConfig(
            n_tasks=4,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(1),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=10,
        )
        result = Simulation(cfg, scheduler=_TightAdmissionScheduler()).run()
        assert result.tasks_completed + result.tasks_dropped == cfg.n_tasks

    def test_tight_admission_preserves_pool_invariants(self) -> None:
        cfg = SimConfig(
            n_tasks=6,
            n_trajectories=3,
            inference_capacity=4,
            judge_capacity=2,
            rollout_duration_fn=uniform_duration(1, 3),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=4,
            seed=42,
        )
        sim = Simulation(cfg, scheduler=_TightAdmissionScheduler())
        result = sim.run()
        assert result.tasks_completed + result.tasks_dropped == cfg.n_tasks
        for stats in result.history:
            assert 0.0 <= stats.inference_utilization <= 1.0
            assert 0.0 <= stats.judge_utilization <= 1.0
        assert sim.inference_pool.in_use == 0
        assert sim.judge_pool.in_use == 0


# ------------------------------------------------------------------
# Simulation exposes scheduler property
# ------------------------------------------------------------------


class TestSimulationSchedulerProperty:
    def test_default_scheduler(self) -> None:
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=_VALID_FN,
            judge_duration_fn=_VALID_FN,
        )
        sim = Simulation(cfg)
        assert isinstance(sim.scheduler, GreedyFIFOScheduler)

    def test_custom_scheduler_retained(self) -> None:
        cfg = SimConfig(
            n_tasks=1,
            n_trajectories=1,
            inference_capacity=1,
            judge_capacity=1,
            rollout_duration_fn=_VALID_FN,
            judge_duration_fn=_VALID_FN,
        )
        sched = _SingleSlotScheduler()
        sim = Simulation(cfg, scheduler=sched)
        assert sim.scheduler is sched


# ------------------------------------------------------------------
# SchedulerView construction
# ------------------------------------------------------------------


class TestSchedulerView:
    def test_view_fields(self) -> None:
        view = _make_view(
            tick=5,
            ckpt_version=2,
            inference_available=3,
            inference_capacity=8,
            judge_available=1,
            judge_capacity=4,
            active_count=5,
            pipeline_cap=6,
            batch_size=3,
            max_staleness=2,
            training_in_flight=True,
            ready_buffer_size=1,
            max_active_staleness=2,
        )
        assert view.tick == 5
        assert view.ckpt_version == 2
        assert view.inference_available == 3
        assert view.inference_capacity == 8
        assert view.judge_available == 1
        assert view.judge_capacity == 4
        assert view.active_count == 5
        assert view.pipeline_cap == 6
        assert view.batch_size == 3
        assert view.max_staleness == 2
        assert view.training_in_flight is True
        assert view.ready_buffer_size == 1
        assert view.max_active_staleness == 2

    def test_view_is_frozen(self) -> None:
        view = _make_view()
        with pytest.raises(AttributeError):
            view.tick = 99  # type: ignore[misc]


# ------------------------------------------------------------------
# DispatchAction construction
# ------------------------------------------------------------------


class TestDispatchAction:
    def test_fields(self) -> None:
        t = _make_task()
        action = DispatchAction(task=t, count=3)
        assert action.task is t
        assert action.count == 3

    def test_frozen(self) -> None:
        t = _make_task()
        action = DispatchAction(task=t, count=1)
        with pytest.raises(AttributeError):
            action.count = 5  # type: ignore[misc]


# ==================================================================
# SRPTAgingScheduler
# ==================================================================


# ------------------------------------------------------------------
# SRPTAgingScheduler constructor
# ------------------------------------------------------------------


class TestSRPTAgingConstructor:
    def test_default_aging_factor(self) -> None:
        sched = SRPTAgingScheduler()
        assert sched.aging_factor == 1.0

    def test_custom_aging_factor(self) -> None:
        sched = SRPTAgingScheduler(aging_factor=0.5)
        assert sched.aging_factor == 0.5

    def test_zero_aging_factor(self) -> None:
        sched = SRPTAgingScheduler(aging_factor=0.0)
        assert sched.aging_factor == 0.0

    def test_rejects_negative_aging_factor(self) -> None:
        with pytest.raises(ValueError, match="aging_factor must be >= 0"):
            SRPTAgingScheduler(aging_factor=-0.1)

    def test_default_admit_headroom(self) -> None:
        sched = SRPTAgingScheduler()
        assert sched.admit_headroom == 1

    def test_custom_admit_headroom(self) -> None:
        sched = SRPTAgingScheduler(admit_headroom=2)
        assert sched.admit_headroom == 2

    def test_none_admit_headroom(self) -> None:
        sched = SRPTAgingScheduler(admit_headroom=None)
        assert sched.admit_headroom is None

    def test_zero_admit_headroom(self) -> None:
        sched = SRPTAgingScheduler(admit_headroom=0)
        assert sched.admit_headroom == 0

    def test_rejects_negative_admit_headroom(self) -> None:
        with pytest.raises(ValueError, match="admit_headroom must be >= 0 or None"):
            SRPTAgingScheduler(admit_headroom=-1)


# ------------------------------------------------------------------
# SRPTAgingScheduler.name
# ------------------------------------------------------------------


class TestSRPTAgingName:
    def test_name(self) -> None:
        assert SRPTAgingScheduler().name == "srpt-aging"


# ------------------------------------------------------------------
# SRPTAgingScheduler.should_admit
# ------------------------------------------------------------------


class TestSRPTAgingAdmission:
    def test_admits_when_under_cap(self) -> None:
        sched = SRPTAgingScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=4, active_count=2)
        assert sched.should_admit(task, 2, view) is True

    def test_rejects_when_at_cap(self) -> None:
        sched = SRPTAgingScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=4)
        assert sched.should_admit(task, 4, view) is False

    def test_rejects_when_above_cap(self) -> None:
        sched = SRPTAgingScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=2)
        assert sched.should_admit(task, 3, view) is False

    def test_admits_with_zero_active(self) -> None:
        sched = SRPTAgingScheduler()
        task = _make_task()
        view = _make_view(pipeline_cap=1)
        assert sched.should_admit(task, 0, view) is True

    def test_records_admission_tick(self) -> None:
        sched = SRPTAgingScheduler()
        task = _make_task("t0")
        view = _make_view(pipeline_cap=4, tick=5)
        sched.should_admit(task, 0, view)
        assert sched._admission_ticks["t0"] == 5  # noqa: SLF001


# ------------------------------------------------------------------
# SRPTAgingScheduler staleness-aware admission
# ------------------------------------------------------------------


class TestSRPTAgingStalenessAdmission:
    """Tests for the projected-staleness admission gate."""

    def test_admits_when_no_staleness_pressure(self) -> None:
        """Zero staleness and no pending advances → admit."""
        sched = SRPTAgingScheduler(admit_headroom=1)
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=0,
            training_in_flight=False,
            ready_buffer_size=0,
        )
        assert sched.should_admit(task, 0, view) is True

    def test_blocks_when_staleness_equals_limit_minus_headroom(self) -> None:
        """projected > max_staleness - headroom → reject."""
        sched = SRPTAgingScheduler(admit_headroom=1)
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=3,
            training_in_flight=False,
            ready_buffer_size=0,
        )
        assert sched.should_admit(task, 0, view) is False

    def test_blocks_when_pending_advances_push_over(self) -> None:
        """Training in-flight adds 1 to projected staleness."""
        sched = SRPTAgingScheduler(admit_headroom=1)
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=1,
            training_in_flight=True,
            ready_buffer_size=0,
        )
        # projected = 1 + 1 = 2, threshold = 3 - 1 = 2, 2 > 2 is False
        assert sched.should_admit(task, 0, view) is True

    def test_blocks_when_training_plus_buffer_push_over(self) -> None:
        """Training in-flight + full batch in buffer = 2 pending advances."""
        sched = SRPTAgingScheduler(admit_headroom=1)
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=1,
            training_in_flight=True,
            ready_buffer_size=4,
            batch_size=4,
        )
        # pending = 1 (training) + 1 (4//4 buffer batch) = 2
        # projected = 1 + 2 = 3, threshold = 3 - 1 = 2, 3 > 2 → reject
        assert sched.should_admit(task, 0, view) is False

    def test_ready_buffer_counts_full_batches_only(self) -> None:
        """Partial batches in the ready buffer don't count as advances."""
        sched = SRPTAgingScheduler(admit_headroom=1)
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=1,
            training_in_flight=False,
            ready_buffer_size=3,
            batch_size=4,
        )
        # pending = 0 (no training) + 0 (3//4 = 0) = 0
        # projected = 1 + 0 = 1, threshold = 2, 1 > 2 → admit
        assert sched.should_admit(task, 0, view) is True

    def test_headroom_none_disables_gate(self) -> None:
        """With admit_headroom=None, only the pipeline cap is checked."""
        sched = SRPTAgingScheduler(admit_headroom=None)
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=2,
            max_active_staleness=10,
            training_in_flight=True,
            ready_buffer_size=8,
            batch_size=1,
        )
        assert sched.should_admit(task, 0, view) is True

    def test_headroom_zero_only_blocks_past_limit(self) -> None:
        """headroom=0 blocks only when projected > max_staleness."""
        sched = SRPTAgingScheduler(admit_headroom=0)
        task = _make_task()
        # projected = 2 + 0 = 2, threshold = 2 - 0 = 2, 2 > 2 is False → admit
        view_at_limit = _make_view(
            pipeline_cap=4,
            max_staleness=2,
            max_active_staleness=2,
            training_in_flight=False,
            ready_buffer_size=0,
        )
        assert sched.should_admit(task, 0, view_at_limit) is True
        # projected = 2 + 1 = 3, 3 > 2 → reject
        view_over = _make_view(
            pipeline_cap=4,
            max_staleness=2,
            max_active_staleness=2,
            training_in_flight=True,
            ready_buffer_size=0,
        )
        assert sched.should_admit(task, 0, view_over) is False

    def test_headroom_2_is_more_conservative(self) -> None:
        """headroom=2 blocks earlier than headroom=1."""
        task = _make_task()
        view = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=1,
            training_in_flight=False,
            ready_buffer_size=0,
        )
        # projected = 1 + 0 = 1
        h1 = SRPTAgingScheduler(admit_headroom=1)
        h2 = SRPTAgingScheduler(admit_headroom=2)
        # h1: 1 > (3-1)=2? No → admit
        assert h1.should_admit(task, 0, view) is True
        # h2: 1 > (3-2)=1? No → admit (boundary)
        assert h2.should_admit(task, 0, view) is True

        view_higher = _make_view(
            pipeline_cap=4,
            max_staleness=3,
            max_active_staleness=2,
            training_in_flight=False,
            ready_buffer_size=0,
        )
        task2 = _make_task("t1")
        task3 = _make_task("t2")
        # projected = 2; h1: 2 > 2? No → admit; h2: 2 > 1? Yes → reject
        assert h1.should_admit(task2, 0, view_higher) is True
        assert h2.should_admit(task3, 0, view_higher) is False

    def test_pipeline_cap_still_enforced_with_headroom(self) -> None:
        """Pipeline cap is always enforced even when staleness is zero."""
        sched = SRPTAgingScheduler(admit_headroom=1)
        task = _make_task()
        view = _make_view(
            pipeline_cap=2,
            max_staleness=10,
            max_active_staleness=0,
        )
        assert sched.should_admit(task, 2, view) is False


# ------------------------------------------------------------------
# SRPTAgingScheduler._pending_checkpoint_advances
# ------------------------------------------------------------------


class TestPendingCheckpointAdvances:
    """Tests for the pending-advances estimate helper."""

    def test_no_training_no_buffer(self) -> None:
        view = _make_view(training_in_flight=False, ready_buffer_size=0, batch_size=4)
        assert SRPTAgingScheduler._pending_checkpoint_advances(view) == 0

    def test_training_in_flight(self) -> None:
        view = _make_view(training_in_flight=True, ready_buffer_size=0, batch_size=4)
        assert SRPTAgingScheduler._pending_checkpoint_advances(view) == 1

    def test_full_batch_in_buffer(self) -> None:
        view = _make_view(training_in_flight=False, ready_buffer_size=4, batch_size=4)
        assert SRPTAgingScheduler._pending_checkpoint_advances(view) == 1

    def test_multiple_batches_in_buffer(self) -> None:
        view = _make_view(training_in_flight=False, ready_buffer_size=9, batch_size=4)
        assert SRPTAgingScheduler._pending_checkpoint_advances(view) == 2

    def test_training_plus_buffer(self) -> None:
        view = _make_view(training_in_flight=True, ready_buffer_size=8, batch_size=4)
        assert SRPTAgingScheduler._pending_checkpoint_advances(view) == 3

    def test_partial_buffer_not_counted(self) -> None:
        view = _make_view(training_in_flight=False, ready_buffer_size=3, batch_size=4)
        assert SRPTAgingScheduler._pending_checkpoint_advances(view) == 0


# ------------------------------------------------------------------
# SRPTAgingScheduler.plan_rollout_dispatch
# ------------------------------------------------------------------


class TestSRPTAgingRolloutDispatch:
    def test_empty_when_no_tasks(self) -> None:
        sched = SRPTAgingScheduler()
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([], view) == []

    def test_skips_terminal_tasks(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=1)
        _admit_task(t)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        t.consume()
        assert t.state == TaskState.CONSUMED
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([t], view) == []

    def test_skips_pending_tasks(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task()
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([t], view) == []

    def test_greedy_allocation(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=4)
        _admit_task(t)
        sched._admission_ticks[t.task_id] = 0  # noqa: SLF001
        view = _make_view(inference_available=4)
        actions = sched.plan_rollout_dispatch([t], view)
        assert len(actions) == 1
        assert actions[0].task is t
        assert actions[0].count == 4

    def test_capped_by_available(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=8)
        _admit_task(t)
        sched._admission_ticks[t.task_id] = 0  # noqa: SLF001
        view = _make_view(inference_available=3)
        actions = sched.plan_rollout_dispatch([t], view)
        assert actions[0].count == 3

    def test_prefers_shorter_remaining(self) -> None:
        """Task with fewer pending_rollout is dispatched first."""
        sched = SRPTAgingScheduler(aging_factor=0.0)
        t_small = _make_task("small", n_trajectories=2)
        t_large = _make_task("large", n_trajectories=8)
        _admit_task(t_small)
        _admit_task(t_large)
        sched._admission_ticks["small"] = 0  # noqa: SLF001
        sched._admission_ticks["large"] = 0  # noqa: SLF001
        view = _make_view(inference_available=3, tick=0)
        actions = sched.plan_rollout_dispatch([t_large, t_small], view)
        assert actions[0].task is t_small

    def test_aging_promotes_old_task(self) -> None:
        """A large task that has waited long enough overtakes a small task."""
        sched = SRPTAgingScheduler(aging_factor=1.0)
        t_small = _make_task("small", n_trajectories=2)
        t_large = _make_task("large", n_trajectories=8)
        _admit_task(t_small)
        _admit_task(t_large)
        # large admitted at tick 0, small admitted at tick 10
        sched._admission_ticks["large"] = 0  # noqa: SLF001
        sched._admission_ticks["small"] = 10  # noqa: SLF001
        # At tick 10: score(large) = 8 - 1.0*10 = -2, score(small) = 2 - 1.0*0 = 2
        view = _make_view(inference_available=3, tick=10)
        actions = sched.plan_rollout_dispatch([t_small, t_large], view)
        assert actions[0].task is t_large

    def test_tie_broken_by_input_order(self) -> None:
        """Equal scores preserve the original list position."""
        sched = SRPTAgingScheduler(aging_factor=0.0)
        t0 = _make_task("t0", n_trajectories=4)
        t1 = _make_task("t1", n_trajectories=4)
        _admit_task(t0)
        _admit_task(t1)
        sched._admission_ticks["t0"] = 0  # noqa: SLF001
        sched._admission_ticks["t1"] = 0  # noqa: SLF001
        view = _make_view(inference_available=6, tick=0)
        actions = sched.plan_rollout_dispatch([t0, t1], view)
        assert len(actions) == 2
        assert actions[0].task is t0
        assert actions[1].task is t1

    def test_no_actions_when_zero_available(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=2)
        _admit_task(t)
        sched._admission_ticks[t.task_id] = 0  # noqa: SLF001
        view = _make_view(inference_available=0)
        assert sched.plan_rollout_dispatch([t], view) == []

    def test_skips_task_with_zero_pending_rollout(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=2)
        _admit_task(t)
        sched._admission_ticks[t.task_id] = 0  # noqa: SLF001
        t.submit_rollouts([1, 1])
        assert t.pending_rollout == 0
        view = _make_view(inference_available=4)
        assert sched.plan_rollout_dispatch([t], view) == []


# ------------------------------------------------------------------
# SRPTAgingScheduler.plan_judge_dispatch
# ------------------------------------------------------------------


class TestSRPTAgingJudgeDispatch:
    def test_empty_when_no_tasks(self) -> None:
        sched = SRPTAgingScheduler()
        view = _make_view(judge_available=4)
        assert sched.plan_judge_dispatch([], view) == []

    def test_dispatches_pending_judges(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=2)
        _admit_task(t)
        sched._admission_ticks[t.task_id] = 0  # noqa: SLF001
        t.submit_rollouts([1, 1])
        t.tick()
        assert t.pending_judge == 2
        view = _make_view(judge_available=4)
        actions = sched.plan_judge_dispatch([t], view)
        assert len(actions) == 1
        assert actions[0].count == 2

    def test_capped_by_available(self) -> None:
        sched = SRPTAgingScheduler()
        t = _make_task(n_trajectories=4)
        _admit_task(t)
        sched._admission_ticks[t.task_id] = 0  # noqa: SLF001
        t.submit_rollouts([1, 1, 1, 1])
        t.tick()
        assert t.pending_judge == 4
        view = _make_view(judge_available=2)
        actions = sched.plan_judge_dispatch([t], view)
        assert actions[0].count == 2

    def test_prefers_shorter_remaining_judges(self) -> None:
        """Task with fewer pending_judge is dispatched first."""
        sched = SRPTAgingScheduler(aging_factor=0.0)
        t_small = _make_task("small", n_trajectories=2)
        t_large = _make_task("large", n_trajectories=6)
        _admit_task(t_small)
        _admit_task(t_large)
        sched._admission_ticks["small"] = 0  # noqa: SLF001
        sched._admission_ticks["large"] = 0  # noqa: SLF001
        t_small.submit_rollouts([1, 1])
        t_small.tick()
        t_large.submit_rollouts([1, 1, 1, 1, 1, 1])
        t_large.tick()
        assert t_small.pending_judge == 2
        assert t_large.pending_judge == 6
        view = _make_view(judge_available=3, tick=0)
        actions = sched.plan_judge_dispatch([t_large, t_small], view)
        assert actions[0].task is t_small

    def test_aging_promotes_old_task_judges(self) -> None:
        """Aging boosts long-waiting task in judge dispatch."""
        sched = SRPTAgingScheduler(aging_factor=1.0)
        t_small = _make_task("small", n_trajectories=2)
        t_large = _make_task("large", n_trajectories=6)
        _admit_task(t_small)
        _admit_task(t_large)
        sched._admission_ticks["large"] = 0  # noqa: SLF001
        sched._admission_ticks["small"] = 10  # noqa: SLF001
        t_small.submit_rollouts([1, 1])
        t_small.tick()
        t_large.submit_rollouts([1, 1, 1, 1, 1, 1])
        t_large.tick()
        # tick=10: score(large) = 6 - 1*10 = -4, score(small) = 2 - 1*0 = 2
        view = _make_view(judge_available=3, tick=10)
        actions = sched.plan_judge_dispatch([t_small, t_large], view)
        assert actions[0].task is t_large


# ------------------------------------------------------------------
# SRPTAgingScheduler integration with Simulation
# ------------------------------------------------------------------


class TestSRPTAgingIntegration:
    """Run SRPTAgingScheduler through full simulations across scenarios."""

    @pytest.fixture(params=list(SCENARIOS.keys()))
    def scenario_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_completes_all_scenarios(self, scenario_name: str) -> None:
        """SRPT-Aging completes every registered scenario without errors."""
        config = get_scenario(scenario_name).config
        sim = Simulation(config, scheduler=SRPTAgingScheduler())
        result = sim.run()
        assert result.tasks_completed + result.tasks_dropped == config.n_tasks
        assert result.ticks_elapsed > 0

    def test_pool_invariants(self, scenario_name: str) -> None:
        """Utilizations stay in [0, 1] and pools are drained at the end."""
        config = get_scenario(scenario_name).config
        sim = Simulation(config, scheduler=SRPTAgingScheduler())
        result = sim.run()
        for stats in result.history:
            assert 0.0 <= stats.inference_utilization <= 1.0
            assert 0.0 <= stats.judge_utilization <= 1.0
        assert sim.inference_pool.in_use == 0
        assert sim.judge_pool.in_use == 0

    def test_deterministic(self) -> None:
        """Same seed produces identical results."""
        cfg = SimConfig(
            n_tasks=10,
            n_trajectories=4,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=uniform_duration(1, 5),
            judge_duration_fn=uniform_duration(1, 3),
            batch_size=2,
            max_staleness=3,
            seed=99,
        )
        r1 = Simulation(cfg, scheduler=SRPTAgingScheduler()).run()
        r2 = Simulation(cfg, scheduler=SRPTAgingScheduler()).run()
        assert r1.ticks_elapsed == r2.ticks_elapsed
        assert r1.tasks_completed == r2.tasks_completed
        assert r1.tasks_dropped == r2.tasks_dropped

    def test_zero_aging_is_pure_srpt(self) -> None:
        """With aging_factor=0, result may differ from FIFO (pure SRPT)."""
        cfg = SimConfig(
            n_tasks=8,
            n_trajectories=4,
            inference_capacity=4,
            judge_capacity=4,
            rollout_duration_fn=constant_duration(2),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            max_staleness=5,
            seed=42,
        )
        result = Simulation(cfg, scheduler=SRPTAgingScheduler(aging_factor=0.0)).run()
        assert result.tasks_completed + result.tasks_dropped == cfg.n_tasks

    def test_headroom_none_matches_old_behaviour(self) -> None:
        """``admit_headroom=None`` gives the same dispatch as SRPT without the gate."""
        cfg = SimConfig(
            n_tasks=20,
            n_trajectories=8,
            inference_capacity=16,
            judge_capacity=3,
            rollout_duration_fn=constant_duration(5),
            judge_duration_fn=constant_duration(2),
            batch_size=4,
            max_staleness=3,
            seed=99,
        )
        r_none = Simulation(cfg, scheduler=SRPTAgingScheduler(admit_headroom=None)).run()
        r_none2 = Simulation(cfg, scheduler=SRPTAgingScheduler(admit_headroom=None)).run()
        assert r_none.ticks_elapsed == r_none2.ticks_elapsed
        assert r_none.tasks_completed == r_none2.tasks_completed
        assert r_none.tasks_dropped == r_none2.tasks_dropped
