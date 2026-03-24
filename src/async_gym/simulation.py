"""Discrete-event simulation runner for the async RL scheduling pipeline.

Orchestrates :class:`Task` state machines and :class:`ReplicaPool` capacity
constraints through a greedy dispatch loop.  Each simulation tick follows three
phases:

1. **Dispatch** — greedily submit rollouts and judges for active tasks, bounded
   by pool capacity.
2. **Tick** — advance all in-flight work by one tick, releasing pool slots for
   completed items.
3. **Consume** — auto-consume any task that has reached ``READY`` (training is
   excluded for now).

Duration distributions for rollouts and judges are pluggable via callable
factories (see :func:`constant_duration` and :func:`uniform_duration`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from async_gym.replica_pool import ReplicaPool
from async_gym.task import Task, TaskState

__all__ = [
    "SimConfig",
    "TickStats",
    "SimResult",
    "Simulation",
    "constant_duration",
    "describe_duration_fn",
    "uniform_duration",
]

DurationFn = Callable[[int, np.random.Generator], list[int]]


def describe_duration_fn(fn: DurationFn) -> str:
    """Return a human-readable description of a duration function.

    Duration factories like :func:`constant_duration` and
    :func:`uniform_duration` attach a ``description`` attribute to the
    returned callable.  This helper reads that attribute, falling back to
    ``"<custom>"`` for user-supplied functions.

    Args:
        fn: A duration function.

    Returns:
        A short description string, e.g. ``"constant(5)"`` or
        ``"uniform(2, 10)"``.
    """
    return getattr(fn, "description", "<custom>")


# ------------------------------------------------------------------
# Duration helper factories
# ------------------------------------------------------------------


def constant_duration(value: int) -> DurationFn:
    """Return a duration function that always produces *value* for every item.

    Args:
        value: The constant tick-duration (must be >= 1).

    Returns:
        A callable ``(n, rng) -> list[int]`` returning ``[value] * n``.

    Raises:
        ValueError: If *value* < 1.

    Examples:
        >>> fn = constant_duration(3)
        >>> fn(4, np.random.default_rng(0))
        [3, 3, 3, 3]
    """
    if value < 1:
        raise ValueError(f"constant duration must be >= 1, got {value}")

    def _sample(n: int, _rng: np.random.Generator) -> list[int]:
        return [value] * n

    _sample.description = f"constant({value})"  # type: ignore[attr-defined]
    return _sample


def uniform_duration(low: int, high: int) -> DurationFn:
    """Return a duration function that samples uniformly from ``[low, high]``.

    Args:
        low: Minimum tick-duration (inclusive, must be >= 1).
        high: Maximum tick-duration (inclusive, must be >= *low*).

    Returns:
        A callable ``(n, rng) -> list[int]`` returning *n* samples drawn from
        ``rng.integers(low, high, endpoint=True)``.

    Raises:
        ValueError: If *low* < 1 or *high* < *low*.

    Examples:
        >>> fn = uniform_duration(2, 5)
        >>> durations = fn(3, np.random.default_rng(42))
        >>> all(2 <= d <= 5 for d in durations)
        True
    """
    if low < 1:
        raise ValueError(f"low must be >= 1, got {low}")
    if high < low:
        raise ValueError(f"high must be >= low ({low}), got {high}")

    def _sample(n: int, rng: np.random.Generator) -> list[int]:
        return rng.integers(low, high, size=n, endpoint=True).tolist()

    _sample.description = f"uniform({low}, {high})"  # type: ignore[attr-defined]
    return _sample


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a single simulation run.

    Args:
        n_tasks: Number of tasks to simulate (>= 1).
        n_trajectories: Trajectories required per task (>= 1).
        inference_capacity: Slots in the inference replica pool (>= 1).
        judge_capacity: Slots in the judge replica pool (>= 1).
        rollout_duration_fn: ``(n, rng) -> list[int]`` producing rollout durations.
        judge_duration_fn: ``(n, rng) -> list[int]`` producing judge durations.
        max_ticks: Safety limit on simulation length (>= 1).
        seed: Optional RNG seed for reproducibility.
    """

    n_tasks: int
    n_trajectories: int
    inference_capacity: int
    judge_capacity: int
    rollout_duration_fn: DurationFn
    judge_duration_fn: DurationFn
    max_ticks: int = 100_000
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate all configuration values."""
        _check_positive("n_tasks", self.n_tasks)
        _check_positive("n_trajectories", self.n_trajectories)
        _check_positive("inference_capacity", self.inference_capacity)
        _check_positive("judge_capacity", self.judge_capacity)
        _check_positive("max_ticks", self.max_ticks)


# ------------------------------------------------------------------
# Per-tick and aggregate result types
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TickStats:
    """Snapshot of a single simulation tick.

    Args:
        tick: The tick number (0-indexed).
        rollouts_dispatched: Rollouts submitted to tasks this tick.
        judges_dispatched: Judges submitted to tasks this tick.
        rollouts_completed: Rollouts that finished this tick.
        judges_completed: Judges that finished this tick.
        tasks_consumed: Tasks that were consumed (moved to CONSUMED) this tick.
        inference_utilization: Fraction of inference slots in use after dispatch.
        judge_utilization: Fraction of judge slots in use after dispatch.
        tasks_by_state: Count of tasks in each :class:`TaskState`.
    """

    tick: int
    rollouts_dispatched: int
    judges_dispatched: int
    rollouts_completed: int
    judges_completed: int
    tasks_consumed: int
    inference_utilization: float
    judge_utilization: float
    tasks_by_state: dict[TaskState, int]


@dataclass(frozen=True)
class SimResult:
    """Aggregate outcome of a completed simulation.

    Args:
        ticks_elapsed: Total ticks the simulation ran.
        tasks_completed: Number of tasks that reached ``CONSUMED``.
        history: Per-tick :class:`TickStats` snapshots.
    """

    ticks_elapsed: int
    tasks_completed: int
    history: list[TickStats] = field(repr=False)


# ------------------------------------------------------------------
# Simulation runner
# ------------------------------------------------------------------


class Simulation:
    """Discrete-event simulation of the async RL scheduling pipeline.

    Creates all tasks up front and runs a greedy dispatch loop until every task
    is consumed or ``max_ticks`` is reached.

    Args:
        config: Simulation parameters.

    Examples:
        >>> cfg = SimConfig(
        ...     n_tasks=2, n_trajectories=2,
        ...     inference_capacity=4, judge_capacity=2,
        ...     rollout_duration_fn=constant_duration(1),
        ...     judge_duration_fn=constant_duration(1),
        ... )
        >>> sim = Simulation(cfg)
        >>> result = sim.run()
        >>> result.tasks_completed
        2
    """

    def __init__(self, config: SimConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

        self._inference_pool = ReplicaPool(name="inference", capacity=config.inference_capacity)
        self._judge_pool = ReplicaPool(name="judge", capacity=config.judge_capacity)

        self._tasks: list[Task] = [
            Task(task_id=f"task-{i}", n_trajectories=config.n_trajectories)
            for i in range(config.n_tasks)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> SimConfig:
        """The simulation configuration."""
        return self._config

    @property
    def tasks(self) -> list[Task]:
        """All tasks in the simulation (read-only reference)."""
        return self._tasks

    @property
    def inference_pool(self) -> ReplicaPool:
        """The inference replica pool."""
        return self._inference_pool

    @property
    def judge_pool(self) -> ReplicaPool:
        """The judge replica pool."""
        return self._judge_pool

    def run(self) -> SimResult:
        """Execute the simulation loop.

        Returns:
            A :class:`SimResult` with the total ticks, tasks completed, and
            per-tick history.
        """
        history: list[TickStats] = []
        tasks_completed = 0

        for tick in range(self._config.max_ticks):
            if tasks_completed == self._config.n_tasks:
                break

            rollouts_dispatched, judges_dispatched = self._dispatch_tick()

            inference_util = self._inference_pool.in_use / self._inference_pool.capacity
            judge_util = self._judge_pool.in_use / self._judge_pool.capacity

            rollouts_completed, judges_completed = self._advance_tick()
            consumed_this_tick = self._consume_ready()
            tasks_completed += consumed_this_tick

            state_counts: dict[TaskState, int] = dict(Counter(t.state for t in self._tasks))

            history.append(
                TickStats(
                    tick=tick,
                    rollouts_dispatched=rollouts_dispatched,
                    judges_dispatched=judges_dispatched,
                    rollouts_completed=rollouts_completed,
                    judges_completed=judges_completed,
                    tasks_consumed=consumed_this_tick,
                    inference_utilization=inference_util,
                    judge_utilization=judge_util,
                    tasks_by_state=state_counts,
                )
            )

        return SimResult(
            ticks_elapsed=len(history),
            tasks_completed=tasks_completed,
            history=history,
        )

    # ------------------------------------------------------------------
    # Per-tick phases
    # ------------------------------------------------------------------

    def _dispatch_tick(self) -> tuple[int, int]:
        """Phase 1: greedily dispatch rollouts and judges for active tasks.

        Iterates tasks in creation order.  For each non-consumed task, submits
        as many rollouts / judges as both the task and the corresponding pool
        allow.

        Returns:
            ``(rollouts_dispatched, judges_dispatched)`` totals for this tick.
        """
        total_rollouts = 0
        total_judges = 0

        for task in self._tasks:
            if task.state == TaskState.CONSUMED:
                continue

            can_rollout = min(task.pending_rollout, self._inference_pool.available)
            if can_rollout > 0:
                durations = self._config.rollout_duration_fn(can_rollout, self._rng)
                task.submit_rollouts(durations)
                self._inference_pool.acquire(can_rollout)
                total_rollouts += can_rollout

            can_judge = min(task.pending_judge, self._judge_pool.available)
            if can_judge > 0:
                durations = self._config.judge_duration_fn(can_judge, self._rng)
                task.submit_judges(durations)
                self._judge_pool.acquire(can_judge)
                total_judges += can_judge

        return total_rollouts, total_judges

    def _advance_tick(self) -> tuple[int, int]:
        """Phase 2: tick all active tasks and release completed pool slots.

        Returns:
            ``(rollouts_completed, judges_completed)`` totals for this tick.
        """
        total_rollouts = 0
        total_judges = 0

        for task in self._tasks:
            if task.state == TaskState.CONSUMED:
                continue

            result = task.tick()

            if result.rollouts_completed > 0:
                self._inference_pool.release(result.rollouts_completed)
                total_rollouts += result.rollouts_completed

            if result.judges_completed > 0:
                self._judge_pool.release(result.judges_completed)
                total_judges += result.judges_completed

        return total_rollouts, total_judges

    def _consume_ready(self) -> int:
        """Phase 3: auto-consume any task in the ``READY`` state.

        Returns:
            Number of tasks consumed this tick.
        """
        consumed = 0
        for task in self._tasks:
            if task.state == TaskState.READY:
                task.consume()
                consumed += 1
        return consumed


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _check_positive(name: str, value: int) -> None:
    """Raise :class:`ValueError` if *value* is not a positive integer.

    Args:
        name: Parameter name (for the error message).
        value: The value to check.
    """
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
