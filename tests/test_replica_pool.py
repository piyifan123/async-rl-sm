"""Tests for the ReplicaPool capacity tracker."""

from __future__ import annotations

import pytest

from async_gym.replica_pool import ReplicaPool

# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_initial_state(self) -> None:
        pool = ReplicaPool(name="inference", capacity=8)
        assert pool.name == "inference"
        assert pool.capacity == 8
        assert pool.in_use == 0
        assert pool.available == 8

    def test_capacity_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            ReplicaPool(name="bad", capacity=0)

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            ReplicaPool(name="bad", capacity=-1)


# ------------------------------------------------------------------
# Acquire
# ------------------------------------------------------------------


class TestAcquire:
    def test_acquire_reduces_available(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        pool.acquire(3)
        assert pool.in_use == 3
        assert pool.available == 1

    def test_acquire_all(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        pool.acquire(4)
        assert pool.in_use == 4
        assert pool.available == 0

    def test_acquire_incremental(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        pool.acquire(1)
        pool.acquire(2)
        assert pool.in_use == 3
        assert pool.available == 1

    def test_acquire_overflow_raises(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        pool.acquire(3)
        with pytest.raises(ValueError, match="only.*available"):
            pool.acquire(2)

    def test_acquire_zero_raises(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        with pytest.raises(ValueError, match="positive"):
            pool.acquire(0)

    def test_acquire_negative_raises(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        with pytest.raises(ValueError, match="positive"):
            pool.acquire(-1)


# ------------------------------------------------------------------
# Release
# ------------------------------------------------------------------


class TestRelease:
    def test_release_increases_available(self) -> None:
        pool = ReplicaPool(name="j", capacity=4)
        pool.acquire(4)
        pool.release(2)
        assert pool.in_use == 2
        assert pool.available == 2

    def test_release_all(self) -> None:
        pool = ReplicaPool(name="j", capacity=4)
        pool.acquire(4)
        pool.release(4)
        assert pool.in_use == 0
        assert pool.available == 4

    def test_release_more_than_in_use_raises(self) -> None:
        pool = ReplicaPool(name="j", capacity=4)
        pool.acquire(2)
        with pytest.raises(ValueError, match="only.*in use"):
            pool.release(3)

    def test_release_when_empty_raises(self) -> None:
        pool = ReplicaPool(name="j", capacity=4)
        with pytest.raises(ValueError, match="only.*in use"):
            pool.release(1)

    def test_release_zero_raises(self) -> None:
        pool = ReplicaPool(name="j", capacity=4)
        pool.acquire(1)
        with pytest.raises(ValueError, match="positive"):
            pool.release(0)

    def test_release_negative_raises(self) -> None:
        pool = ReplicaPool(name="j", capacity=4)
        pool.acquire(1)
        with pytest.raises(ValueError, match="positive"):
            pool.release(-1)


# ------------------------------------------------------------------
# Acquire / release cycle
# ------------------------------------------------------------------


class TestAcquireReleaseCycle:
    def test_full_cycle(self) -> None:
        pool = ReplicaPool(name="inf", capacity=4)
        pool.acquire(4)
        assert pool.available == 0
        pool.release(2)
        assert pool.available == 2
        pool.acquire(1)
        assert pool.available == 1
        pool.release(3)
        assert pool.available == 4
        assert pool.in_use == 0

    def test_interleaved_with_task_tick(self) -> None:
        """Simulate the scheduler pattern: acquire on submit, release on tick."""
        from async_gym.task import Task

        pool = ReplicaPool(name="inference", capacity=3)
        t = Task(task_id="t0", n_trajectories=5)

        # Submit 3 rollouts (fills pool)
        can_submit = min(t.pending_rollout, pool.available)
        assert can_submit == 3
        t.submit_rollouts([1, 2, 3])
        pool.acquire(3)
        assert pool.available == 0

        # Tick: 1-tick rollout completes, frees a slot
        result = t.tick()
        pool.release(result.rollouts_completed)
        assert result.rollouts_completed == 1
        assert pool.available == 1

        # Submit 1 more rollout into the freed slot
        t.submit_rollouts([1])
        pool.acquire(1)
        assert pool.available == 0

        # Tick: the 1-tick rollout and the 2-tick rollout complete
        result = t.tick()
        pool.release(result.rollouts_completed)
        assert result.rollouts_completed == 2
        assert pool.available == 2


# ------------------------------------------------------------------
# Invariant
# ------------------------------------------------------------------


class TestInvariant:
    def test_invariant_holds_through_operations(self) -> None:
        pool = ReplicaPool(name="test", capacity=10)

        def check() -> None:
            assert 0 <= pool.in_use <= pool.capacity
            assert pool.available == pool.capacity - pool.in_use

        check()
        pool.acquire(5)
        check()
        pool.release(3)
        check()
        pool.acquire(8)
        check()
        pool.release(10)
        check()


# ------------------------------------------------------------------
# Repr
# ------------------------------------------------------------------


class TestRepr:
    def test_name_and_capacity_in_repr(self) -> None:
        pool = ReplicaPool(name="judge", capacity=6)
        r = repr(pool)
        assert "judge" in r
        assert "6" in r

    def test_in_use_hidden_in_repr(self) -> None:
        pool = ReplicaPool(name="judge", capacity=6)
        assert "_in_use" not in repr(pool)
