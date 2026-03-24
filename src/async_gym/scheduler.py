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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from async_gym.task import Task, TaskState

__all__ = [
    "DispatchAction",
    "GreedyFIFOScheduler",
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
