"""Task state machine for asynchronous off-policy RL scheduling.

Each task requires *N* trajectories to be rolled out and judged before it is
ready for trainer consumption.  The task's high-level state is derived from
trajectory-stage counters so that state and counters can never drift apart.

In-flight stages (rollout and judging) are tracked as lists of :class:`InFlight`
items, each with its own tick-based duration, enabling simulation of variable
rollout lengths and judge latencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar

__all__ = ["TaskState", "Task", "InFlight", "TickResult"]


class TaskState(Enum):
    """Observable lifecycle state of an RL task.

    The ordering follows the natural pipeline progression:

        PENDING → ROLLING_OUT → PARTIAL → JUDGING → READY → CONSUMED
    """

    PENDING = auto()
    ROLLING_OUT = auto()
    PARTIAL = auto()
    JUDGING = auto()
    READY = auto()
    CONSUMED = auto()


@dataclass
class InFlight:
    """A single trajectory currently being processed (rollout or judging).

    Progress is tracked in discrete *ticks*.  Call :meth:`tick` once per
    simulation step; the item is done when ``remaining_ticks`` reaches zero.

    Args:
        total_ticks: Total duration of this work item in ticks (must be ≥ 1).

    Examples:
        >>> item = InFlight(total_ticks=3)
        >>> item.tick()
        False
        >>> item.tick(); item.tick()
        True
        >>> item.done
        True
    """

    total_ticks: int
    elapsed_ticks: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.total_ticks < 1:
            raise ValueError(f"total_ticks must be >= 1, got {self.total_ticks}")

    @property
    def remaining_ticks(self) -> int:
        """Number of ticks until this item completes."""
        return self.total_ticks - self.elapsed_ticks

    @property
    def done(self) -> bool:
        """Whether this item has finished processing."""
        return self.elapsed_ticks >= self.total_ticks

    def tick(self) -> bool:
        """Advance by one tick.

        Returns:
            ``True`` if the item *just* completed on this tick, ``False``
            otherwise (already done or still in progress).
        """
        if self.done:
            return False
        self.elapsed_ticks += 1
        return self.done


@dataclass(frozen=True)
class TickResult:
    """Outcome of a single :meth:`Task.tick` call.

    Args:
        rollouts_completed: Number of rollouts that finished on this tick.
        judges_completed: Number of judge calls that finished on this tick.
    """

    rollouts_completed: int
    judges_completed: int


@dataclass
class Task:
    """State machine for a single RL task.

    A task tracks *N* trajectories across five pipeline stages.  The three
    counter stages (``pending_rollout``, ``pending_judge``, ``judged``) are
    plain integers.  The two in-flight stages are lists of :class:`InFlight`
    items, enabling per-trajectory duration simulation via :meth:`tick`.

    The high-level ``state`` property is derived from these counters/lists.

    Args:
        task_id: Unique identifier for this task.
        n_trajectories: Number of trajectories required (global *N*).

    Examples:
        >>> t = Task(task_id="t0", n_trajectories=4)
        >>> t.submit_rollouts([3, 5, 2, 4])
        >>> t.state
        <TaskState.ROLLING_OUT: ...>
        >>> result = t.tick()
    """

    task_id: str
    n_trajectories: int

    pending_rollout: int = field(init=False, default=0)
    _rollouts_in_flight: list[InFlight] = field(init=False, default_factory=list, repr=False)
    pending_judge: int = field(init=False, default=0)
    _judges_in_flight: list[InFlight] = field(init=False, default_factory=list, repr=False)
    judged: int = field(init=False, default=0)

    _consumed: bool = field(init=False, default=False, repr=False)
    birth_ckpt: int | None = field(init=False, default=None, repr=False)

    _VALID_N_RANGE: ClassVar[range] = range(1, 10_001)

    def __post_init__(self) -> None:
        if self.n_trajectories not in self._VALID_N_RANGE:
            raise ValueError(
                f"n_trajectories must be in [{self._VALID_N_RANGE.start}, "
                f"{self._VALID_N_RANGE.stop - 1}], got {self.n_trajectories}"
            )
        self.pending_rollout = self.n_trajectories

    # ------------------------------------------------------------------
    # Convenience counter properties
    # ------------------------------------------------------------------

    @property
    def rolling_out(self) -> int:
        """Number of trajectories currently in rollout."""
        return len(self._rollouts_in_flight)

    @property
    def judging(self) -> int:
        """Number of trajectories currently being judged."""
        return len(self._judges_in_flight)

    @property
    def rollouts_in_flight(self) -> Sequence[InFlight]:
        """Read-only view of in-flight rollout items."""
        return self._rollouts_in_flight

    @property
    def judges_in_flight(self) -> Sequence[InFlight]:
        """Read-only view of in-flight judge items."""
        return self._judges_in_flight

    def staleness(self, current_ckpt: int) -> int:
        """Compute how stale this task's data is relative to the current checkpoint.

        Staleness is the number of training steps that have occurred since the
        task first started rolling out (i.e. ``current_ckpt - birth_ckpt``).

        Args:
            current_ckpt: The global checkpoint version to measure against.

        Returns:
            The non-negative staleness count.

        Raises:
            ValueError: If :attr:`birth_ckpt` has not been set yet (task is
                still ``PENDING``).
        """
        if self.birth_ckpt is None:
            raise ValueError(
                f"Cannot compute staleness for task {self.task_id!r}: birth_ckpt has not been set"
            )
        return current_ckpt - self.birth_ckpt

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def state(self) -> TaskState:
        """Compute the current task state from trajectory counters.

        Returns:
            The ``TaskState`` that matches the current counter distribution.
        """
        if self._consumed:
            return TaskState.CONSUMED
        if self.judged == self.n_trajectories:
            return TaskState.READY

        rollout_has_work = self.pending_rollout > 0 or self.rolling_out > 0
        judge_has_work = self.pending_judge > 0 or self.judging > 0

        if not rollout_has_work and judge_has_work:
            return TaskState.JUDGING
        if rollout_has_work and (judge_has_work or self.judged > 0):
            return TaskState.PARTIAL
        if self.pending_rollout < self.n_trajectories:
            return TaskState.ROLLING_OUT
        return TaskState.PENDING

    # ------------------------------------------------------------------
    # Invariant
    # ------------------------------------------------------------------

    def _assert_invariant(self) -> None:
        """Verify the counter-sum invariant.

        Raises:
            AssertionError: If counters do not sum to ``n_trajectories``.
        """
        total = (
            self.pending_rollout
            + self.rolling_out
            + self.pending_judge
            + self.judging
            + self.judged
        )
        assert total == self.n_trajectories, (
            f"Invariant violated: counters sum to {total}, expected {self.n_trajectories}"
        )

    # ------------------------------------------------------------------
    # Transition interface
    # ------------------------------------------------------------------

    def submit_rollouts(self, durations: Sequence[int]) -> None:
        """Move trajectories from ``pending_rollout`` to in-flight rollout.

        Each entry in *durations* specifies the tick-duration for one trajectory.

        Args:
            durations: Tick-durations for each trajectory to dispatch.  Length
                determines how many trajectories are moved.

        Raises:
            ValueError: If the task is consumed, *durations* is empty, or there
                are not enough pending trajectories.
        """
        self._check_not_consumed("submit_rollouts")
        k = len(durations)
        if k == 0:
            raise ValueError("durations must be non-empty")
        if k > self.pending_rollout:
            raise ValueError(f"Cannot submit {k} rollouts (only {self.pending_rollout} pending)")
        self.pending_rollout -= k
        self._rollouts_in_flight.extend(InFlight(total_ticks=d) for d in durations)
        self._assert_invariant()

    def submit_judges(self, durations: Sequence[int]) -> None:
        """Move trajectories from ``pending_judge`` to in-flight judging.

        Each entry in *durations* specifies the tick-duration for one judge call.

        Args:
            durations: Tick-durations for each trajectory to dispatch for
                judging.  Length determines how many trajectories are moved.

        Raises:
            ValueError: If the task is consumed, *durations* is empty, or there
                are not enough pending-judge trajectories.
        """
        self._check_not_consumed("submit_judges")
        k = len(durations)
        if k == 0:
            raise ValueError("durations must be non-empty")
        if k > self.pending_judge:
            raise ValueError(f"Cannot submit {k} judges (only {self.pending_judge} pending)")
        self.pending_judge -= k
        self._judges_in_flight.extend(InFlight(total_ticks=d) for d in durations)
        self._assert_invariant()

    def tick(self) -> TickResult:
        """Advance all in-flight work by one tick.

        Trajectories that complete their rollout are moved to ``pending_judge``.
        Trajectories that complete judging are moved to ``judged``.

        Returns:
            A :class:`TickResult` summarising how many items completed.

        Raises:
            ValueError: If the task has been consumed.
        """
        self._check_not_consumed("tick")

        # Tick all rollouts and partition into done / still going.
        rollouts_completed = 0
        still_rolling: list[InFlight] = []
        for item in self._rollouts_in_flight:
            item.tick()
            if item.done:
                rollouts_completed += 1
            else:
                still_rolling.append(item)
        self._rollouts_in_flight[:] = still_rolling
        self.pending_judge += rollouts_completed

        # Tick all judges and partition into done / still going.
        judges_completed = 0
        still_judging: list[InFlight] = []
        for item in self._judges_in_flight:
            item.tick()
            if item.done:
                judges_completed += 1
            else:
                still_judging.append(item)
        self._judges_in_flight[:] = still_judging
        self.judged += judges_completed

        self._assert_invariant()
        return TickResult(
            rollouts_completed=rollouts_completed,
            judges_completed=judges_completed,
        )

    def consume(self) -> None:
        """Mark the task as consumed by the trainer.

        Raises:
            ValueError: If the task is not in the ``READY`` state.
        """
        if self.state != TaskState.READY:
            raise ValueError(
                f"Cannot consume task {self.task_id!r} in state {self.state.name}; "
                "task must be READY"
            )
        self._consumed = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_not_consumed(self, method: str) -> None:
        """Raise if the task has already been consumed.

        Args:
            method: Name of the calling method (for the error message).

        Raises:
            ValueError: If the task is in ``CONSUMED`` state.
        """
        if self._consumed:
            raise ValueError(f"Cannot call {method}() on consumed task {self.task_id!r}")
