"""Discrete-event simulation runner for the async RL scheduling pipeline.

Orchestrates :class:`Task` state machines and :class:`ReplicaPool` capacity
constraints through a pluggable dispatch scheduler.  Each simulation tick
follows five phases:

1. **Dispatch** — the :class:`~async_gym.scheduler.Scheduler` admits PENDING
   tasks, then plans rollout and judge slot allocation.  The simulation
   validates and executes the plan.
2. **Record utilisation** — snapshot pool occupancy after dispatch.
3. **Advance** — tick all in-flight task work *and* any in-progress training,
   releasing pool slots for completed items and incrementing the global
   checkpoint when training finishes.
4. **Collect ready** — move newly-READY tasks into a buffer awaiting training.
5. **Training gate** — if no training is running and enough tasks are buffered,
   consume exactly ``batch_size`` tasks and start a training run.

Duration distributions for rollouts and judges are pluggable via callable
factories (see :func:`constant_duration` and :func:`uniform_duration`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from async_gym.replica_pool import ReplicaPool
from async_gym.scheduler import (
    DispatchAction,
    GreedyFIFOScheduler,
    Scheduler,
    SchedulerView,
)
from async_gym.task import InFlight, Task, TaskState

__all__ = [
    "SimConfig",
    "TickStats",
    "SimResult",
    "Simulation",
    "bimodal_duration",
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


def bimodal_duration(short: int, long: int, p_short: float = 0.8) -> DurationFn:
    """Return a duration function that draws from a two-point mixture.

    Each sampled duration is *short* with probability *p_short* and *long*
    otherwise.  This produces high coefficient-of-variation workloads that
    stress scheduling algorithms — exactly the regime where SRPT-family
    policies outperform FIFO.

    Args:
        short: Duration (in ticks) for the "easy" mode (must be >= 1).
        long: Duration (in ticks) for the "hard" mode (must be > *short*).
        p_short: Probability of drawing the short duration (0 < p_short < 1).

    Returns:
        A callable ``(n, rng) -> list[int]`` returning *n* samples from the
        mixture.

    Raises:
        ValueError: If *short* < 1, *long* <= *short*, or *p_short* is not
            in the open interval (0, 1).

    Examples:
        >>> fn = bimodal_duration(2, 40, p_short=0.8)
        >>> durations = fn(1000, np.random.default_rng(0))
        >>> set(durations) == {2, 40}
        True
    """
    if short < 1:
        raise ValueError(f"short must be >= 1, got {short}")
    if long <= short:
        raise ValueError(f"long must be > short ({short}), got {long}")
    if not (0 < p_short < 1):
        raise ValueError(f"p_short must be in (0, 1), got {p_short}")

    def _sample(n: int, rng: np.random.Generator) -> list[int]:
        choices = rng.random(n) < p_short
        return [short if c else long for c in choices]

    _sample.description = f"bimodal({short}, {long}, p={p_short})"  # type: ignore[attr-defined]
    return _sample


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a single simulation run.

    Args:
        n_tasks: Number of tasks to simulate (>= 1, must be a multiple of
            *batch_size*).
        n_trajectories: Trajectories required per task (>= 1).
        inference_capacity: Slots in the inference replica pool (>= 1).
        judge_capacity: Slots in the judge replica pool (>= 1).
        rollout_duration_fn: ``(n, rng) -> list[int]`` producing rollout durations.
        judge_duration_fn: ``(n, rng) -> list[int]`` producing judge durations.
        batch_size: Number of READY tasks consumed per training run (>= 1).
        training_speed: Divisor applied to compute training duration (>= 1.0).
            Higher values simulate faster training hardware.  Training
            duration in ticks is
            ``ceil(sampled_rollout_dur * n_trajectories * batch_size * 3 / training_speed)``.
        max_staleness: Maximum checkpoint-distance allowed between a task's
            birth checkpoint and the checkpoint at consumption (>= 1).
        max_ticks: Safety limit on simulation length (>= 1).
        seed: Optional RNG seed for reproducibility.
    """

    n_tasks: int
    n_trajectories: int
    inference_capacity: int
    judge_capacity: int
    rollout_duration_fn: DurationFn
    judge_duration_fn: DurationFn
    batch_size: int = 1
    training_speed: float = 1.0
    max_staleness: int = 1
    max_ticks: int = 100_000
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate all configuration values."""
        _check_positive("n_tasks", self.n_tasks)
        _check_positive("n_trajectories", self.n_trajectories)
        _check_positive("inference_capacity", self.inference_capacity)
        _check_positive("judge_capacity", self.judge_capacity)
        _check_positive("batch_size", self.batch_size)
        _check_positive("max_staleness", self.max_staleness)
        _check_positive("max_ticks", self.max_ticks)
        if self.training_speed < 1.0:
            raise ValueError(f"training_speed must be >= 1.0, got {self.training_speed}")
        if self.n_tasks % self.batch_size != 0:
            raise ValueError(
                f"n_tasks ({self.n_tasks}) must be a multiple of batch_size ({self.batch_size})"
            )


# ------------------------------------------------------------------
# Per-tick and aggregate result types
# ------------------------------------------------------------------


_TERMINAL_STATES = frozenset({TaskState.CONSUMED, TaskState.DROPPED})


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
        tasks_dropped: Tasks that were dropped (too stale) this tick.
        inference_utilization: Fraction of inference slots in use after dispatch.
        judge_utilization: Fraction of judge slots in use after dispatch.
        tasks_by_state: Count of tasks in each :class:`TaskState`.
        training_active: Whether a training run is in flight this tick.
        training_utilization: Fraction of trainer capacity in use (1.0 when
            training, 0.0 when idle).
        training_ticks_total: Total duration (in ticks) of the current
            training run, or 0 if no training is in flight.
        ckpt_version: Global checkpoint version at the end of this tick.
        ready_buffer_size: Number of READY tasks waiting for training.
        max_task_staleness: Maximum staleness across all active (non-PENDING,
            non-CONSUMED, non-DROPPED) tasks, or 0 if there are none.
    """

    tick: int
    rollouts_dispatched: int
    judges_dispatched: int
    rollouts_completed: int
    judges_completed: int
    tasks_consumed: int
    tasks_dropped: int
    inference_utilization: float
    judge_utilization: float
    tasks_by_state: dict[TaskState, int]
    training_active: bool
    training_utilization: float
    training_ticks_total: int
    ckpt_version: int
    ready_buffer_size: int
    max_task_staleness: int


@dataclass(frozen=True)
class SimResult:
    """Aggregate outcome of a completed simulation.

    Args:
        ticks_elapsed: Total ticks the simulation ran.
        tasks_completed: Number of tasks that reached ``CONSUMED``.
        tasks_dropped: Number of tasks that were dropped (too stale at
            consumption time).
        history: Per-tick :class:`TickStats` snapshots.
    """

    ticks_elapsed: int
    tasks_completed: int
    tasks_dropped: int
    history: list[TickStats] = field(repr=False)


# ------------------------------------------------------------------
# Simulation runner
# ------------------------------------------------------------------


class Simulation:
    """Discrete-event simulation of the async RL scheduling pipeline.

    Creates all tasks up front and runs a five-phase dispatch loop until every
    task is finished (consumed or dropped) or ``max_ticks`` is reached.
    Training is batch-gated: ``batch_size`` READY tasks must accumulate before
    a training run starts.  A pipeline-depth throttle limits the number of
    active tasks to ``max_staleness * batch_size``.  At consumption time, any
    task whose staleness exceeds ``max_staleness`` is dropped instead of
    consumed.

    Dispatch scheduling is delegated to a :class:`~async_gym.scheduler.Scheduler`
    instance, which controls admission, rollout allocation, and judge allocation
    each tick.  The default is :class:`~async_gym.scheduler.GreedyFIFOScheduler`.

    Args:
        config: Simulation parameters.
        scheduler: Dispatch scheduler.  Defaults to
            :class:`~async_gym.scheduler.GreedyFIFOScheduler`.

    Examples:
        >>> cfg = SimConfig(
        ...     n_tasks=2, n_trajectories=2,
        ...     inference_capacity=4, judge_capacity=2,
        ...     rollout_duration_fn=constant_duration(1),
        ...     judge_duration_fn=constant_duration(1),
        ...     batch_size=2,
        ... )
        >>> sim = Simulation(cfg)
        >>> result = sim.run()
        >>> result.tasks_completed
        2
    """

    def __init__(self, config: SimConfig, scheduler: Scheduler | None = None) -> None:
        self._config = config
        self._scheduler = scheduler or GreedyFIFOScheduler()
        self._rng = np.random.default_rng(config.seed)

        self._inference_pool = ReplicaPool(name="inference", capacity=config.inference_capacity)
        self._judge_pool = ReplicaPool(name="judge", capacity=config.judge_capacity)

        self._tasks: list[Task] = [
            Task(task_id=f"task-{i}", n_trajectories=config.n_trajectories)
            for i in range(config.n_tasks)
        ]

        self._ckpt_version: int = 0
        self._current_tick: int = 0
        self._training_in_flight: InFlight | None = None
        self._ready_buffer: list[Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> SimConfig:
        """The simulation configuration."""
        return self._config

    @property
    def scheduler(self) -> Scheduler:
        """The dispatch scheduler."""
        return self._scheduler

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

    @property
    def ckpt_version(self) -> int:
        """The current global checkpoint version."""
        return self._ckpt_version

    def run(self) -> SimResult:
        """Execute the simulation loop.

        Returns:
            A :class:`SimResult` with the total ticks, tasks completed/dropped,
            and per-tick history.
        """
        history: list[TickStats] = []
        tasks_completed = 0
        tasks_dropped = 0

        for tick in range(self._config.max_ticks):
            if tasks_completed + tasks_dropped == self._config.n_tasks:
                break

            self._current_tick = tick

            # Phase 1 — dispatch via scheduler
            rollouts_dispatched, judges_dispatched = self._dispatch_tick()

            # Phase 2 — record utilisation
            inference_util = self._inference_pool.in_use / self._inference_pool.capacity
            judge_util = self._judge_pool.in_use / self._judge_pool.capacity

            # Phase 3 — advance tasks and training
            rollouts_completed, judges_completed = self._advance_tick()

            # Phase 4 — collect newly-READY tasks into buffer
            self._collect_ready()

            # Phase 5 — training gate (with staleness enforcement)
            consumed_this_tick, dropped_this_tick = self._maybe_start_training()
            tasks_completed += consumed_this_tick
            tasks_dropped += dropped_this_tick

            # Record stats
            state_counts: dict[TaskState, int] = dict(Counter(t.state for t in self._tasks))
            max_staleness = self._current_max_staleness()
            is_training = self._training_in_flight is not None
            trn_total = self._training_in_flight.total_ticks if is_training else 0

            history.append(
                TickStats(
                    tick=tick,
                    rollouts_dispatched=rollouts_dispatched,
                    judges_dispatched=judges_dispatched,
                    rollouts_completed=rollouts_completed,
                    judges_completed=judges_completed,
                    tasks_consumed=consumed_this_tick,
                    tasks_dropped=dropped_this_tick,
                    inference_utilization=inference_util,
                    judge_utilization=judge_util,
                    tasks_by_state=state_counts,
                    training_active=is_training,
                    training_utilization=1.0 if is_training else 0.0,
                    training_ticks_total=trn_total,
                    ckpt_version=self._ckpt_version,
                    ready_buffer_size=len(self._ready_buffer),
                    max_task_staleness=max_staleness,
                )
            )

        return SimResult(
            ticks_elapsed=len(history),
            tasks_completed=tasks_completed,
            tasks_dropped=tasks_dropped,
            history=history,
        )

    # ------------------------------------------------------------------
    # Per-tick phases
    # ------------------------------------------------------------------

    def _dispatch_tick(self) -> tuple[int, int]:
        """Phase 1: dispatch rollouts and judges via the scheduler.

        Builds a :class:`SchedulerView`, runs the admission phase, then asks
        the scheduler for rollout and judge allocation plans.  Plans are
        validated and executed.

        Returns:
            ``(rollouts_dispatched, judges_dispatched)`` totals for this tick.
        """
        view = self._build_scheduler_view()

        # --- Admission phase ---
        active_count = view.active_count
        for task in self._tasks:
            if task.birth_ckpt is not None or task.state in _TERMINAL_STATES:
                continue
            if self._scheduler.should_admit(task, active_count, view):
                task.birth_ckpt = self._ckpt_version
                active_count += 1

        # --- Dispatch phase (both plans from the same snapshot) ---
        rollout_plan = self._scheduler.plan_rollout_dispatch(self._tasks, view)
        judge_plan = self._scheduler.plan_judge_dispatch(self._tasks, view)

        self._validate_rollout_plan(rollout_plan, view)
        self._validate_judge_plan(judge_plan, view)

        total_rollouts = self._execute_rollout_plan(rollout_plan)
        total_judges = self._execute_judge_plan(judge_plan)

        return total_rollouts, total_judges

    def _build_scheduler_view(self) -> SchedulerView:
        """Create a read-only snapshot of current simulation state.

        Returns:
            A frozen :class:`SchedulerView` for the scheduler hooks.
        """
        cfg = self._config
        active = sum(
            1 for t in self._tasks if t.birth_ckpt is not None and t.state not in _TERMINAL_STATES
        )
        return SchedulerView(
            tick=self._current_tick,
            ckpt_version=self._ckpt_version,
            inference_available=self._inference_pool.available,
            inference_capacity=self._inference_pool.capacity,
            judge_available=self._judge_pool.available,
            judge_capacity=self._judge_pool.capacity,
            active_count=active,
            pipeline_cap=cfg.max_staleness * cfg.batch_size,
            batch_size=cfg.batch_size,
            max_staleness=cfg.max_staleness,
            training_in_flight=self._training_in_flight is not None,
            ready_buffer_size=len(self._ready_buffer),
            max_active_staleness=self._current_max_staleness(),
        )

    def _validate_rollout_plan(
        self,
        actions: Sequence[DispatchAction],
        view: SchedulerView,
    ) -> None:
        """Raise if the rollout plan violates capacity or per-task constraints.

        Args:
            actions: Rollout dispatch actions from the scheduler.
            view: The scheduler view snapshot used for planning.

        Raises:
            ValueError: On any constraint violation.
        """
        total = 0
        for action in actions:
            task = action.task
            if task.state in _TERMINAL_STATES:
                raise ValueError(
                    f"Rollout plan includes terminal task {task.task_id!r} "
                    f"(state={task.state.name})"
                )
            if task.birth_ckpt is None:
                raise ValueError(f"Rollout plan includes non-admitted task {task.task_id!r}")
            if action.count < 1:
                raise ValueError(
                    f"Rollout action for {task.task_id!r} has count={action.count}; must be >= 1"
                )
            if action.count > task.pending_rollout:
                raise ValueError(
                    f"Rollout action for {task.task_id!r}: count={action.count} "
                    f"exceeds pending_rollout={task.pending_rollout}"
                )
            total += action.count
        if total > view.inference_available:
            raise ValueError(
                f"Rollout plan total={total} exceeds inference_available={view.inference_available}"
            )

    def _validate_judge_plan(
        self,
        actions: Sequence[DispatchAction],
        view: SchedulerView,
    ) -> None:
        """Raise if the judge plan violates capacity or per-task constraints.

        Args:
            actions: Judge dispatch actions from the scheduler.
            view: The scheduler view snapshot used for planning.

        Raises:
            ValueError: On any constraint violation.
        """
        total = 0
        for action in actions:
            task = action.task
            if task.state in _TERMINAL_STATES:
                raise ValueError(
                    f"Judge plan includes terminal task {task.task_id!r} (state={task.state.name})"
                )
            if task.birth_ckpt is None:
                raise ValueError(f"Judge plan includes non-admitted task {task.task_id!r}")
            if action.count < 1:
                raise ValueError(
                    f"Judge action for {task.task_id!r} has count={action.count}; must be >= 1"
                )
            if action.count > task.pending_judge:
                raise ValueError(
                    f"Judge action for {task.task_id!r}: count={action.count} "
                    f"exceeds pending_judge={task.pending_judge}"
                )
            total += action.count
        if total > view.judge_available:
            raise ValueError(
                f"Judge plan total={total} exceeds judge_available={view.judge_available}"
            )

    def _execute_rollout_plan(self, actions: Sequence[DispatchAction]) -> int:
        """Execute validated rollout dispatch actions.

        Args:
            actions: Validated rollout dispatch actions.

        Returns:
            Total rollout slots dispatched.
        """
        total = 0
        for action in actions:
            durations = self._config.rollout_duration_fn(action.count, self._rng)
            action.task.submit_rollouts(durations)
            self._inference_pool.acquire(action.count)
            total += action.count
        return total

    def _execute_judge_plan(self, actions: Sequence[DispatchAction]) -> int:
        """Execute validated judge dispatch actions.

        Args:
            actions: Validated judge dispatch actions.

        Returns:
            Total judge slots dispatched.
        """
        total = 0
        for action in actions:
            durations = self._config.judge_duration_fn(action.count, self._rng)
            action.task.submit_judges(durations)
            self._judge_pool.acquire(action.count)
            total += action.count
        return total

    def _advance_tick(self) -> tuple[int, int]:
        """Phase 3: tick all active tasks and training, release pool slots.

        If a training run completes this tick, the global checkpoint version
        is incremented.

        Returns:
            ``(rollouts_completed, judges_completed)`` totals for this tick.
        """
        total_rollouts = 0
        total_judges = 0

        for task in self._tasks:
            if task.state in _TERMINAL_STATES:
                continue
            if task.birth_ckpt is None:
                continue

            result = task.tick()

            if result.rollouts_completed > 0:
                self._inference_pool.release(result.rollouts_completed)
                total_rollouts += result.rollouts_completed

            if result.judges_completed > 0:
                self._judge_pool.release(result.judges_completed)
                total_judges += result.judges_completed

        if self._training_in_flight is not None:
            self._training_in_flight.tick()
            if self._training_in_flight.done:
                self._ckpt_version += 1
                self._training_in_flight = None

        return total_rollouts, total_judges

    def _collect_ready(self) -> None:
        """Phase 4: move newly-READY tasks into the ready buffer.

        Tasks that have just become READY (all trajectories judged, not yet
        consumed, and not already buffered) are appended to ``_ready_buffer``
        in creation order.
        """
        buffered_ids = {t.task_id for t in self._ready_buffer}
        for task in self._tasks:
            if task.state == TaskState.READY and task.task_id not in buffered_ids:
                self._ready_buffer.append(task)

    def _maybe_start_training(self) -> tuple[int, int]:
        """Phase 5: start a training run if the batch is full and trainer idle.

        Pops ``batch_size`` tasks from the front of the ready buffer.  Each
        task's staleness is checked: tasks exceeding ``max_staleness`` are
        dropped, the rest are consumed.  Training starts only if at least one
        task was consumed.

        Returns:
            ``(consumed, dropped)`` counts for this tick.
        """
        cfg = self._config
        if self._training_in_flight is not None:
            return 0, 0
        if len(self._ready_buffer) < cfg.batch_size:
            return 0, 0

        batch = self._ready_buffer[: cfg.batch_size]
        self._ready_buffer = self._ready_buffer[cfg.batch_size :]

        consumed = 0
        dropped = 0
        for task in batch:
            if task.staleness(self._ckpt_version) > cfg.max_staleness:
                task.drop()
                dropped += 1
            else:
                task.consume()
                consumed += 1

        if consumed > 0:
            sampled_duration = cfg.rollout_duration_fn(1, self._rng)[0]
            training_ticks = max(
                1,
                math.ceil(
                    sampled_duration * cfg.n_trajectories * cfg.batch_size * 3 / cfg.training_speed
                ),
            )
            self._training_in_flight = InFlight(total_ticks=training_ticks)

        return consumed, dropped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_max_staleness(self) -> int:
        """Return the maximum staleness across all active tasks.

        Returns:
            The highest ``staleness`` value, or 0 if no tasks have started.
        """
        worst = 0
        for task in self._tasks:
            if task.birth_ckpt is not None and task.state not in _TERMINAL_STATES:
                worst = max(worst, task.staleness(self._ckpt_version))
        return worst


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
