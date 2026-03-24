"""Pluggable dispatch-scheduling interface for the simulation runner.

Defines three hook points that control how the simulation allocates resources
each tick:

1. **Admission** (:meth:`Scheduler.should_admit`) — whether a PENDING task
   should enter the active pipeline.
2. **Rollout dispatch** (:meth:`Scheduler.plan_rollout_dispatch`) — how many
   inference slots each active task receives, and in what priority order.
3. **Judge dispatch** (:meth:`Scheduler.plan_judge_dispatch`) — how many judge
   slots each active task receives, and in what priority order.

:class:`GreedyFIFOScheduler` replicates the original hard-coded behaviour:
creation-order iteration, greedy ``min(pending, available)`` allocation, and
pipeline-cap admission.

:class:`SRPTAgingScheduler` prioritises tasks with less remaining work
(Shortest Remaining Processing Time) while applying a linear aging boost
so that long-running tasks are not starved indefinitely.  See
``docs/scheduling.md`` for the full algorithm description and queueing-theory
analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from async_gym.task import Task, TaskState

__all__ = [
    "DispatchAction",
    "GreedyFIFOScheduler",
    "SRPTAgingScheduler",
    "Scheduler",
    "SchedulerView",
]

_TERMINAL_STATES = frozenset({TaskState.CONSUMED, TaskState.DROPPED})


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerView:
    """Read-only snapshot of simulation state provided to scheduler hooks.

    Built once per tick by the simulation before calling the scheduler.

    Args:
        tick: Current tick number (0-indexed).
        ckpt_version: Global checkpoint version.
        inference_available: Free slots in the inference (rollout) pool.
        inference_capacity: Total slots in the inference pool.
        judge_available: Free slots in the judge pool.
        judge_capacity: Total slots in the judge pool.
        active_count: Number of tasks with ``birth_ckpt`` set that have not
            reached a terminal state.
        pipeline_cap: Maximum allowed active tasks
            (``max_staleness * batch_size``).
        batch_size: Tasks consumed per training run.
        max_staleness: Maximum checkpoint distance before a task is dropped.
        training_in_flight: Whether a training run is currently in progress.
        ready_buffer_size: Number of READY tasks waiting in the training
            buffer.
        max_active_staleness: Maximum staleness across all active
            (non-terminal, admitted) tasks, or 0 if there are none.
    """

    tick: int
    ckpt_version: int
    inference_available: int
    inference_capacity: int
    judge_available: int
    judge_capacity: int
    active_count: int
    pipeline_cap: int
    batch_size: int
    max_staleness: int
    training_in_flight: bool
    ready_buffer_size: int
    max_active_staleness: int


@dataclass(frozen=True)
class DispatchAction:
    """A single resource-allocation decision for one task in one queue.

    Returned by :meth:`Scheduler.plan_rollout_dispatch` and
    :meth:`Scheduler.plan_judge_dispatch`.

    Args:
        task: The task to receive resources.
        count: Number of slots to allocate (must be >= 1).
    """

    task: Task
    count: int


# ------------------------------------------------------------------
# Abstract scheduler interface
# ------------------------------------------------------------------


class Scheduler(ABC):
    """Abstract base class for dispatch schedulers.

    Subclass and implement the three abstract methods to create a custom
    scheduling strategy.  The simulation calls these hooks during the
    dispatch phase of each tick.

    See :class:`GreedyFIFOScheduler` for the baseline implementation that
    replicates the original hard-coded behaviour.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this scheduling strategy."""

    @abstractmethod
    def should_admit(
        self,
        task: Task,
        active_count: int,
        view: SchedulerView,
    ) -> bool:
        """Decide whether a PENDING task should enter the active pipeline.

        Called once per PENDING task in creation order.  The *active_count*
        parameter is a live value updated by the simulation as earlier tasks
        are admitted within the same tick.

        Args:
            task: A task in the ``PENDING`` state (``birth_ckpt is None``).
            active_count: Number of currently active (non-terminal, admitted)
                tasks, including any admitted earlier this tick.
            view: Snapshot of simulation state at the start of this tick.

        Returns:
            ``True`` to admit the task (the simulation will set its
            ``birth_ckpt``), ``False`` to skip it this tick.
        """

    @abstractmethod
    def plan_rollout_dispatch(
        self,
        tasks: Sequence[Task],
        view: SchedulerView,
    ) -> Sequence[DispatchAction]:
        """Plan inference-slot allocation for this tick.

        Called after the admission phase.  The returned list order defines
        dispatch priority — the simulation executes actions in sequence.
        Each :class:`DispatchAction` specifies a task and how many rollout
        slots it should receive.

        Args:
            tasks: All tasks in the simulation (including terminal and
                pending).  Implementations should filter appropriately.
            view: Snapshot of simulation state at the start of this tick.

        Returns:
            Ordered sequence of rollout dispatch actions.  Must satisfy:

            - ``sum(a.count) <= view.inference_available``
            - Per action: ``1 <= a.count <= task.pending_rollout``
            - Only non-terminal tasks with ``birth_ckpt`` set
        """

    @abstractmethod
    def plan_judge_dispatch(
        self,
        tasks: Sequence[Task],
        view: SchedulerView,
    ) -> Sequence[DispatchAction]:
        """Plan judge-slot allocation for this tick.

        Same contract as :meth:`plan_rollout_dispatch` but for judge slots.

        Args:
            tasks: All tasks in the simulation.
            view: Snapshot of simulation state at the start of this tick.

        Returns:
            Ordered sequence of judge dispatch actions.  Must satisfy:

            - ``sum(a.count) <= view.judge_available``
            - Per action: ``1 <= a.count <= task.pending_judge``
            - Only non-terminal tasks with ``birth_ckpt`` set
        """


# ------------------------------------------------------------------
# Baseline implementation
# ------------------------------------------------------------------


class GreedyFIFOScheduler(Scheduler):
    """Greedy first-in-first-out scheduler (baseline).

    Replicates the original ``Simulation._dispatch_tick`` behaviour:

    - **Admission**: admit PENDING tasks while ``active_count < pipeline_cap``.
    - **Rollout dispatch**: iterate tasks in creation order, give each
      ``min(pending_rollout, remaining_inference)`` slots.
    - **Judge dispatch**: iterate tasks in creation order, give each
      ``min(pending_judge, remaining_judge)`` slots.
    """

    @property
    def name(self) -> str:
        """Return ``'greedy-fifo'``."""
        return "greedy-fifo"

    def should_admit(
        self,
        task: Task,
        active_count: int,
        view: SchedulerView,
    ) -> bool:
        """Admit while under the pipeline cap.

        Args:
            task: A PENDING task.
            active_count: Live count of active tasks.
            view: Simulation state snapshot.

        Returns:
            ``True`` if ``active_count < view.pipeline_cap``.
        """
        return active_count < view.pipeline_cap

    def plan_rollout_dispatch(
        self,
        tasks: Sequence[Task],
        view: SchedulerView,
    ) -> list[DispatchAction]:
        """Greedy FIFO rollout allocation.

        Iterates tasks in input order.  Each active task with pending
        rollouts receives ``min(pending_rollout, remaining_slots)``.

        Args:
            tasks: All simulation tasks.
            view: Simulation state snapshot.

        Returns:
            Ordered list of rollout dispatch actions.
        """
        actions: list[DispatchAction] = []
        remaining = view.inference_available
        for task in tasks:
            if remaining <= 0:
                break
            if task.state in _TERMINAL_STATES or task.birth_ckpt is None:
                continue
            n = min(task.pending_rollout, remaining)
            if n > 0:
                actions.append(DispatchAction(task=task, count=n))
                remaining -= n
        return actions

    def plan_judge_dispatch(
        self,
        tasks: Sequence[Task],
        view: SchedulerView,
    ) -> list[DispatchAction]:
        """Greedy FIFO judge allocation.

        Iterates tasks in input order.  Each active task with pending
        judges receives ``min(pending_judge, remaining_slots)``.

        Args:
            tasks: All simulation tasks.
            view: Simulation state snapshot.

        Returns:
            Ordered list of judge dispatch actions.
        """
        actions: list[DispatchAction] = []
        remaining = view.judge_available
        for task in tasks:
            if remaining <= 0:
                break
            if task.state in _TERMINAL_STATES or task.birth_ckpt is None:
                continue
            n = min(task.pending_judge, remaining)
            if n > 0:
                actions.append(DispatchAction(task=task, count=n))
                remaining -= n
        return actions


# ------------------------------------------------------------------
# SRPT + Aging implementation
# ------------------------------------------------------------------


class SRPTAgingScheduler(Scheduler):
    """Shortest-Remaining-Processing-Time scheduler with aging.

    Dispatch priority is determined by the score::

        score(task) = pending_work - aging_factor * age_ticks

    where *pending_work* is ``pending_rollout`` (rollout phase) or
    ``pending_judge`` (judge phase), and *age_ticks* is the number of
    ticks since the task was admitted.  Tasks with **lower** scores are
    dispatched first.  Ties are broken by input-list order for stability.

    - ``aging_factor = 0`` yields pure SRPT (prefer short remaining work).
    - Large ``aging_factor`` approaches FIFO (oldest tasks first).

    **Starvation bound**: a task with *W* remaining work overtakes a
    competing task with *w < W* remaining work after at most
    ``(W - w) / aging_factor`` idle ticks.

    **Admission** uses a projected-staleness gate that estimates pending
    checkpoint advances and blocks admission when the oldest active task
    would be pushed past the staleness limit.  Set ``admit_headroom=None``
    to fall back to the simple pipeline-cap gate used by the baseline.

    Args:
        aging_factor: Non-negative weight applied to each tick of age.
            Defaults to ``1.0``.
        admit_headroom: Staleness headroom required for admission.
            The gate blocks when ``projected_staleness >
            max_staleness - admit_headroom``.  ``None`` disables the
            staleness-aware gate entirely (simple pipeline-cap only).
            Defaults to ``1``.

    Raises:
        ValueError: If *aging_factor* is negative or *admit_headroom*
            is negative.
    """

    def __init__(
        self,
        aging_factor: float = 1.0,
        admit_headroom: int | None = 1,
    ) -> None:
        if aging_factor < 0:
            raise ValueError(f"aging_factor must be >= 0, got {aging_factor}")
        if admit_headroom is not None and admit_headroom < 0:
            raise ValueError(f"admit_headroom must be >= 0 or None, got {admit_headroom}")
        self._aging_factor = aging_factor
        self._admit_headroom = admit_headroom
        self._admission_ticks: dict[str, int] = {}

    @property
    def name(self) -> str:
        """Return ``'srpt-aging'``."""
        return "srpt-aging"

    @property
    def aging_factor(self) -> float:
        """The aging weight used in the priority score."""
        return self._aging_factor

    @property
    def admit_headroom(self) -> int | None:
        """Staleness headroom required for admission, or ``None`` if disabled."""
        return self._admit_headroom

    def should_admit(
        self,
        task: Task,
        active_count: int,
        view: SchedulerView,
    ) -> bool:
        """Decide whether to admit a PENDING task using staleness-aware gating.

        The pipeline-cap ceiling is always enforced.  When
        ``admit_headroom`` is not ``None``, a projected-staleness gate
        additionally blocks admission if the oldest active task would be
        pushed past the staleness limit by pending checkpoint advances.

        Args:
            task: A PENDING task.
            active_count: Live count of active tasks.
            view: Simulation state snapshot.

        Returns:
            ``True`` to admit, ``False`` to defer to a later tick.
        """
        if active_count >= view.pipeline_cap:
            return False

        if self._admit_headroom is not None:
            pending = self._pending_checkpoint_advances(view)
            projected = view.max_active_staleness + pending
            if projected > view.max_staleness - self._admit_headroom:
                return False

        self._admission_ticks[task.task_id] = view.tick
        return True

    @staticmethod
    def _pending_checkpoint_advances(view: SchedulerView) -> int:
        """Estimate checkpoint advances already committed in the pipeline.

        Counts in-flight training (will advance the checkpoint on
        completion) plus full batches sitting in the ready buffer.

        Args:
            view: Simulation state snapshot.

        Returns:
            Non-negative count of expected upcoming checkpoint advances.
        """
        advances = 0
        if view.training_in_flight:
            advances += 1
        advances += view.ready_buffer_size // view.batch_size
        return advances

    def _score(self, pending: int, task_id: str, current_tick: int) -> float:
        """Compute the dispatch priority score for a task.

        Lower scores receive slots first.

        Args:
            pending: Remaining work items (rollouts or judges).
            task_id: Identifier used to look up the admission tick.
            current_tick: The current simulation tick.

        Returns:
            ``pending - aging_factor * age_ticks``.
        """
        admission_tick = self._admission_ticks.get(task_id, current_tick)
        age = current_tick - admission_tick
        return pending - self._aging_factor * age

    def _sorted_candidates(
        self,
        tasks: Sequence[Task],
        pending_attr: str,
        current_tick: int,
    ) -> list[tuple[float, int, Task]]:
        """Filter and sort tasks by priority score.

        Args:
            tasks: All simulation tasks.
            pending_attr: ``'pending_rollout'`` or ``'pending_judge'``.
            current_tick: The current simulation tick.

        Returns:
            List of ``(score, original_index, task)`` sorted ascending by
            score then by original index for tie-breaking stability.
        """
        candidates: list[tuple[float, int, Task]] = []
        for idx, task in enumerate(tasks):
            if task.state in _TERMINAL_STATES or task.birth_ckpt is None:
                continue
            pending: int = getattr(task, pending_attr)
            if pending <= 0:
                continue
            score = self._score(pending, task.task_id, current_tick)
            candidates.append((score, idx, task))
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates

    def plan_rollout_dispatch(
        self,
        tasks: Sequence[Task],
        view: SchedulerView,
    ) -> list[DispatchAction]:
        """SRPT-with-aging rollout allocation.

        Tasks are sorted by ``pending_rollout - aging_factor * age``,
        then greedily allocated ``min(pending_rollout, remaining_slots)``.

        Args:
            tasks: All simulation tasks.
            view: Simulation state snapshot.

        Returns:
            Ordered list of rollout dispatch actions.
        """
        actions: list[DispatchAction] = []
        remaining = view.inference_available
        for _score, _idx, task in self._sorted_candidates(tasks, "pending_rollout", view.tick):
            if remaining <= 0:
                break
            n = min(task.pending_rollout, remaining)
            if n > 0:
                actions.append(DispatchAction(task=task, count=n))
                remaining -= n
        return actions

    def plan_judge_dispatch(
        self,
        tasks: Sequence[Task],
        view: SchedulerView,
    ) -> list[DispatchAction]:
        """SRPT-with-aging judge allocation.

        Tasks are sorted by ``pending_judge - aging_factor * age``,
        then greedily allocated ``min(pending_judge, remaining_slots)``.

        Args:
            tasks: All simulation tasks.
            view: Simulation state snapshot.

        Returns:
            Ordered list of judge dispatch actions.
        """
        actions: list[DispatchAction] = []
        remaining = view.judge_available
        for _score, _idx, task in self._sorted_candidates(tasks, "pending_judge", view.tick):
            if remaining <= 0:
                break
            n = min(task.pending_judge, remaining)
            if n > 0:
                actions.append(DispatchAction(task=task, count=n))
                remaining -= n
        return actions
