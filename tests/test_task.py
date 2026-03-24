"""Tests for the Task state machine with tick-based in-flight tracking."""

from __future__ import annotations

import pytest

from async_gym.task import InFlight, Task, TaskState, TickResult

N = 4


# ------------------------------------------------------------------
# InFlight unit tests
# ------------------------------------------------------------------


class TestInFlight:
    def test_initial_state(self) -> None:
        f = InFlight(total_ticks=3)
        assert f.elapsed_ticks == 0
        assert f.remaining_ticks == 3
        assert not f.done

    def test_tick_progression(self) -> None:
        f = InFlight(total_ticks=2)
        assert f.tick() is False  # tick 1, not done yet
        assert f.elapsed_ticks == 1
        assert f.remaining_ticks == 1
        assert f.tick() is True  # tick 2, just completed
        assert f.done

    def test_tick_after_done_is_noop(self) -> None:
        f = InFlight(total_ticks=1)
        assert f.tick() is True
        assert f.tick() is False
        assert f.elapsed_ticks == 1

    def test_single_tick_duration(self) -> None:
        f = InFlight(total_ticks=1)
        assert f.tick() is True
        assert f.done

    def test_zero_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_ticks"):
            InFlight(total_ticks=0)

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_ticks"):
            InFlight(total_ticks=-1)


# ------------------------------------------------------------------
# Construction & initial state
# ------------------------------------------------------------------


class TestConstruction:
    def test_initial_state_is_pending(self) -> None:
        t = Task(task_id="t0", n_trajectories=N)
        assert t.state == TaskState.PENDING

    def test_initial_counters(self) -> None:
        t = Task(task_id="t0", n_trajectories=N)
        assert t.pending_rollout == N
        assert t.rolling_out == 0
        assert t.pending_judge == 0
        assert t.judging == 0
        assert t.judged == 0

    def test_initial_in_flight_lists_empty(self) -> None:
        t = Task(task_id="t0", n_trajectories=N)
        assert len(t.rollouts_in_flight) == 0
        assert len(t.judges_in_flight) == 0

    def test_n_trajectories_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_trajectories"):
            Task(task_id="bad", n_trajectories=0)

    def test_n_trajectories_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="n_trajectories"):
            Task(task_id="bad", n_trajectories=10_001)


# ------------------------------------------------------------------
# Full lifecycle: PENDING → ROLLING_OUT → PARTIAL → JUDGING → READY → CONSUMED
# ------------------------------------------------------------------


class TestFullLifecycle:
    """Walk a task through every state with N=4, using tick-based progression."""

    def test_walk_all_states(self) -> None:
        t = Task(task_id="lc", n_trajectories=N)
        assert t.state == TaskState.PENDING

        # Dispatch 2 rollouts (durations 2 and 3 ticks) → ROLLING_OUT
        t.submit_rollouts([2, 3])
        assert t.state == TaskState.ROLLING_OUT
        assert t.pending_rollout == 2
        assert t.rolling_out == 2

        # Tick 1: nothing completes yet
        r = t.tick()
        assert r == TickResult(rollouts_completed=0, judges_completed=0)
        assert t.state == TaskState.ROLLING_OUT

        # Tick 2: first rollout (duration=2) completes → pending_judge=1
        # pending_rollout=2, pending_judge=1 → PARTIAL
        r = t.tick()
        assert r.rollouts_completed == 1
        assert t.pending_judge == 1
        assert t.state == TaskState.PARTIAL

        # Submit that completed rollout for judging (duration=2), dispatch remaining 2 rollouts
        t.submit_judges([2])
        t.submit_rollouts([1, 1])
        # rolling_out=3, judging=1 → PARTIAL
        assert t.state == TaskState.PARTIAL

        # Tick 3: second original rollout (duration=3, elapsed=2→3) completes,
        # two new rollouts (duration=1) complete, judge not done yet
        r = t.tick()
        assert r.rollouts_completed == 3
        assert r.judges_completed == 0
        assert t.rolling_out == 0
        assert t.pending_rollout == 0
        assert t.pending_judge == 3
        assert t.judging == 1
        assert t.state == TaskState.JUDGING

        # Submit the 3 pending judges (durations 1, 1, 1)
        t.submit_judges([1, 1, 1])
        assert t.state == TaskState.JUDGING

        # Tick 4: the 3 new judges (duration=1) complete, original judge (duration=2) completes
        r = t.tick()
        assert r.judges_completed == 4
        assert t.judged == N
        assert t.state == TaskState.READY

        # Consume
        t.consume()
        assert t.state == TaskState.CONSUMED


# ------------------------------------------------------------------
# State derivation edge cases
# ------------------------------------------------------------------


class TestStateDerived:
    def test_rolling_out_only(self) -> None:
        t = Task(task_id="r", n_trajectories=N)
        t.submit_rollouts([5, 5, 5, 5])
        assert t.state == TaskState.ROLLING_OUT

    def test_partial_via_pending_judge(self) -> None:
        """pending_rollout > 0 AND pending_judge > 0 → PARTIAL."""
        t = Task(task_id="p", n_trajectories=N)
        t.submit_rollouts([1])
        t.tick()  # rollout completes → pending_judge=1
        # pending_rollout=3, pending_judge=1
        assert t.state == TaskState.PARTIAL

    def test_partial_via_judged(self) -> None:
        """pending_rollout > 0 AND judged > 0 → PARTIAL."""
        t = Task(task_id="p2", n_trajectories=N)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        # pending_rollout=3, judged=1
        assert t.state == TaskState.PARTIAL

    def test_judging_state(self) -> None:
        t = Task(task_id="j", n_trajectories=2)
        t.submit_rollouts([1, 1])
        t.tick()  # both rollouts complete
        t.submit_judges([3])
        # pending_rollout=0, rolling_out=0, pending_judge=1, judging=1
        assert t.state == TaskState.JUDGING

    def test_ready_state(self) -> None:
        t = Task(task_id="rdy", n_trajectories=1)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        assert t.state == TaskState.READY


# ------------------------------------------------------------------
# Tick mechanics
# ------------------------------------------------------------------


class TestTick:
    def test_tick_on_empty_task_is_noop(self) -> None:
        t = Task(task_id="noop", n_trajectories=N)
        r = t.tick()
        assert r == TickResult(rollouts_completed=0, judges_completed=0)
        assert t.state == TaskState.PENDING

    def test_variable_durations_complete_at_different_ticks(self) -> None:
        t = Task(task_id="var", n_trajectories=3)
        t.submit_rollouts([1, 3, 5])

        r = t.tick()
        assert r.rollouts_completed == 1
        assert t.rolling_out == 2

        t.tick()
        r = t.tick()
        assert r.rollouts_completed == 1
        assert t.rolling_out == 1

        t.tick()
        r = t.tick()
        assert r.rollouts_completed == 1
        assert t.rolling_out == 0

    def test_rollout_completion_moves_to_pending_judge(self) -> None:
        t = Task(task_id="mv", n_trajectories=2)
        t.submit_rollouts([1, 1])
        t.tick()
        assert t.pending_judge == 2
        assert t.rolling_out == 0

    def test_judge_completion_moves_to_judged(self) -> None:
        t = Task(task_id="jc", n_trajectories=1)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        assert t.judged == 1

    def test_rollout_and_judge_progress_simultaneously(self) -> None:
        """Rollouts and judges in-flight at the same time both advance per tick."""
        t = Task(task_id="sim", n_trajectories=2)
        t.submit_rollouts([1, 3])
        t.tick()  # first rollout completes
        t.submit_judges([2])  # send it to judging

        # Now: rolling_out=1 (2 ticks left), judging=1 (2 ticks left)
        r = t.tick()
        assert r.rollouts_completed == 0
        assert r.judges_completed == 0

        r = t.tick()
        assert r.rollouts_completed == 1  # second rollout done
        assert r.judges_completed == 1  # judge done

    def test_in_flight_items_observable(self) -> None:
        t = Task(task_id="obs", n_trajectories=2)
        t.submit_rollouts([3, 5])
        assert len(t.rollouts_in_flight) == 2
        assert t.rollouts_in_flight[0].total_ticks == 3
        assert t.rollouts_in_flight[1].total_ticks == 5

        t.tick()
        assert t.rollouts_in_flight[0].elapsed_ticks == 1
        assert t.rollouts_in_flight[0].remaining_ticks == 2


# ------------------------------------------------------------------
# Transition validation
# ------------------------------------------------------------------


class TestTransitionErrors:
    def test_submit_rollouts_more_than_available(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        with pytest.raises(ValueError, match="only.*pending"):
            t.submit_rollouts([1] * (N + 1))

    def test_submit_rollouts_empty(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        with pytest.raises(ValueError, match="non-empty"):
            t.submit_rollouts([])

    def test_submit_judges_more_than_pending(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        t.submit_rollouts([1])
        t.tick()
        with pytest.raises(ValueError, match="only.*pending"):
            t.submit_judges([1, 1])

    def test_submit_judges_empty(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        t.submit_rollouts([1])
        t.tick()
        with pytest.raises(ValueError, match="non-empty"):
            t.submit_judges([])

    def test_consume_when_not_ready(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        with pytest.raises(ValueError, match="must be READY"):
            t.consume()

    def test_transition_after_consumed(self) -> None:
        t = Task(task_id="e", n_trajectories=1)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        t.consume()
        with pytest.raises(ValueError, match="consumed"):
            t.submit_rollouts([1])

    def test_tick_after_consumed(self) -> None:
        t = Task(task_id="e", n_trajectories=1)
        t.submit_rollouts([1])
        t.tick()
        t.submit_judges([1])
        t.tick()
        t.consume()
        with pytest.raises(ValueError, match="consumed"):
            t.tick()

    def test_invalid_rollout_duration(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        with pytest.raises(ValueError, match="total_ticks"):
            t.submit_rollouts([0])

    def test_invalid_judge_duration(self) -> None:
        t = Task(task_id="e", n_trajectories=N)
        t.submit_rollouts([1])
        t.tick()
        with pytest.raises(ValueError, match="total_ticks"):
            t.submit_judges([0])


# ------------------------------------------------------------------
# Counter invariant
# ------------------------------------------------------------------


class TestInvariant:
    def test_invariant_holds_through_interleaved_operations(self) -> None:
        t = Task(task_id="inv", n_trajectories=N)

        def assert_sum() -> None:
            total = t.pending_rollout + t.rolling_out + t.pending_judge + t.judging + t.judged
            assert total == N

        assert_sum()

        t.submit_rollouts([2, 3])
        assert_sum()

        t.tick()
        assert_sum()

        t.tick()  # first rollout (dur=2) completes
        assert_sum()

        t.submit_judges([1])
        assert_sum()

        t.submit_rollouts([1, 4])
        assert_sum()

        t.tick()  # second rollout (dur=3) + new rollout (dur=1) + judge (dur=1) complete
        assert_sum()

        t.submit_judges([1, 1])
        assert_sum()

        for _ in range(10):
            t.tick()
            assert_sum()

        # Everything should have completed by now
        assert t.judged + t.pending_judge == N - t.rolling_out - t.pending_rollout - t.judging


# ------------------------------------------------------------------
# Repr / identity
# ------------------------------------------------------------------


class TestRepr:
    def test_task_id_in_repr(self) -> None:
        t = Task(task_id="hello", n_trajectories=2)
        assert "hello" in repr(t)

    def test_consumed_flag_hidden_in_repr(self) -> None:
        t = Task(task_id="x", n_trajectories=1)
        assert "_consumed" not in repr(t)

    def test_in_flight_lists_hidden_in_repr(self) -> None:
        t = Task(task_id="x", n_trajectories=1)
        assert "_rollouts_in_flight" not in repr(t)
        assert "_judges_in_flight" not in repr(t)
