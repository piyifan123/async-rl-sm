#!/usr/bin/env python3
"""Run async-RL scheduling simulation scenarios and print summaries.

Usage::

    uv run run_sim.py                                        # default scenario, greedy-fifo
    uv run run_sim.py --scenario adversarial                 # one scenario by name
    uv run run_sim.py --scenario default adversarial         # several scenarios
    uv run run_sim.py --all                                  # all registered scenarios
    uv run run_sim.py --scheduler greedy-fifo                # specify scheduler by name
    uv run run_sim.py --list                                 # list scenarios and exit
    uv run run_sim.py --list-schedulers                      # list schedulers and exit
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from async_gym.scenarios import Scenario, get_scenario, list_scenarios
from async_gym.scheduler import GreedyFIFOScheduler, Scheduler, SRPTAgingScheduler
from async_gym.simulation import (
    SimResult,
    TickStats,
    describe_duration_fn,
)
from async_gym.task import TaskState

# ------------------------------------------------------------------
# Scheduler registry (name -> factory)
# ------------------------------------------------------------------

SCHEDULERS: dict[str, type[Scheduler]] = {
    "greedy-fifo": GreedyFIFOScheduler,
    "srpt-aging": SRPTAgingScheduler,
}


def _get_scheduler(name: str) -> Scheduler:
    """Look up a scheduler by name and return a fresh instance.

    Args:
        name: Scheduler name (case-sensitive).

    Returns:
        A new :class:`Scheduler` instance.

    Raises:
        KeyError: If *name* is not registered.
    """
    try:
        cls = SCHEDULERS[name]
    except KeyError:
        available = ", ".join(sorted(SCHEDULERS))
        raise KeyError(f"unknown scheduler {name!r}; available: {available}") from None
    return cls()


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

_STATE_ORDER = [
    TaskState.PENDING,
    TaskState.ROLLING_OUT,
    TaskState.PARTIAL,
    TaskState.JUDGING,
    TaskState.READY,
    TaskState.CONSUMED,
    TaskState.DROPPED,
]


def _format_state_distribution(stats: TickStats) -> str:
    """Format the task-state distribution as a readable fixed-width string.

    Args:
        stats: The tick snapshot to format.

    Returns:
        A string like ``pending=2  rolling_out=3  partial=5  ...``.
    """
    labels = {
        "PENDING": "pending",
        "ROLLING_OUT": "rolling_out",
        "PARTIAL": "partial",
        "JUDGING": "judging",
        "READY": "ready",
        "CONSUMED": "consumed",
        "DROPPED": "dropped",
    }
    parts = []
    for state in _STATE_ORDER:
        count = stats.tasks_by_state.get(state, 0)
        parts.append(f"{labels[state.name]}={count}")
    return "  ".join(parts)


def _print_header(scenario: Scenario, scheduler: Scheduler) -> None:
    """Print the scenario description and simulation configuration header.

    Args:
        scenario: The scenario to display.
        scheduler: The scheduler being used.
    """
    cfg = scenario.config
    print("=" * 120)
    print(f"Scenario: {scenario.name}  |  Scheduler: {scheduler.name}")
    wrapped = textwrap.fill(scenario.description, width=114)
    for line in wrapped.splitlines():
        print(f"  {line}")
    print("=" * 120)
    print("Async-RL Scheduling Simulation")
    print("-" * 120)
    print(f"  Tasks:                {cfg.n_tasks}")
    print(f"  Trajectories/task:    {cfg.n_trajectories}")
    print(f"  Inference capacity:   {cfg.inference_capacity}")
    print(f"  Judge capacity:       {cfg.judge_capacity}")
    print(f"  Rollout duration:     {describe_duration_fn(cfg.rollout_duration_fn)}")
    print(f"  Judge duration:       {describe_duration_fn(cfg.judge_duration_fn)}")
    print(f"  Batch size:           {cfg.batch_size}")
    print(
        f"  Training speed:       {cfg.training_speed}"
        f"  (duration = ceil(rollout_sample * n_trajectories * batch_size * 3 / speed))"
    )
    print(f"  Max staleness:        {cfg.max_staleness}")
    print(f"  Max ticks:            {cfg.max_ticks}")
    print(f"  Seed:                 {cfg.seed}")
    print(f"  Scheduler:            {scheduler.name}")
    print("-" * 120)


def _print_snapshots(history: list[TickStats], interval: int) -> None:
    """Print periodic tick snapshots.

    In addition to every *interval*-th tick and the final tick, any tick
    where a task is dropped or consumed is always printed so that key
    events are never hidden by a large interval.

    Args:
        history: Full tick history from the simulation.
        interval: Print every *interval*-th tick plus the final tick.
    """
    print(
        f"\n{'Tick':>6}  {'Inf%':>5}  {'Jdg%':>5}  {'Trn%':>5}  {'TrnDur':>6}  "
        f"{'R_disp':>6}  {'J_disp':>6}  {'R_done':>6}  {'J_done':>6}  "
        f"{'Ckpt':>4}  {'Buf':>3}  {'Stale':>5}  {'Drop':>4}  "
        f"State distribution"
    )
    print("-" * 156)

    for stats in history:
        is_periodic = stats.tick % interval == 0
        is_last = stats.tick == history[-1].tick
        has_event = stats.tasks_dropped > 0 or stats.tasks_consumed > 0
        if not (is_periodic or is_last or has_event):
            continue

        trn_dur = str(stats.training_ticks_total) if stats.training_active else ""
        print(
            f"{stats.tick:>6}  "
            f"{stats.inference_utilization:>5.1%}  "
            f"{stats.judge_utilization:>5.1%}  "
            f"{stats.training_utilization:>5.1%}  "
            f"{trn_dur:>6}  "
            f"{stats.rollouts_dispatched:>6}  "
            f"{stats.judges_dispatched:>6}  "
            f"{stats.rollouts_completed:>6}  "
            f"{stats.judges_completed:>6}  "
            f"{stats.ckpt_version:>4}  "
            f"{stats.ready_buffer_size:>3}  "
            f"{stats.max_task_staleness:>5}  "
            f"{stats.tasks_dropped:>4}  "
            f"{_format_state_distribution(stats)}"
        )


def _print_summary(result: SimResult) -> None:
    """Print the final summary.

    Args:
        result: The simulation result.
    """
    n = len(result.history) if result.history else 1
    avg_inf = sum(s.inference_utilization for s in result.history) / n
    avg_jdg = sum(s.judge_utilization for s in result.history) / n
    avg_trn = sum(s.training_utilization for s in result.history) / n

    final_ckpt = result.history[-1].ckpt_version if result.history else 0
    peak_staleness = max(s.max_task_staleness for s in result.history) if result.history else 0

    total_drops = result.tasks_dropped

    print("\n" + "=" * 120)
    print("Summary")
    print("=" * 120)
    print(f"  Ticks elapsed:        {result.ticks_elapsed}")
    print(f"  Tasks completed:      {result.tasks_completed}")
    print(f"  Tasks dropped:        {total_drops}")
    print(f"  Final checkpoint:     {final_ckpt}")
    print(f"  Peak staleness:       {peak_staleness}")
    print(f"  Avg inference util:   {avg_inf:.1%}")
    print(f"  Avg judge util:       {avg_jdg:.1%}")
    print(f"  Avg training util:    {avg_trn:.1%}")
    print("=" * 120)


def _run_scenario(scenario: Scenario, scheduler: Scheduler) -> None:
    """Run a single scenario and print its full output.

    Args:
        scenario: The scenario to execute.
        scheduler: The scheduler to use for dispatch decisions.
    """
    from async_gym.simulation import Simulation

    sim = Simulation(scenario.config, scheduler=scheduler)
    result = sim.run()

    _print_snapshots(result.history, scenario.snapshot_interval)
    print()
    _print_header(scenario, scheduler)
    _print_summary(result)


def _list_scenarios() -> None:
    """Print a table of all registered scenarios and exit."""
    scenarios = list_scenarios()
    print(f"{'Name':<20}  Description")
    print("-" * 80)
    for s in scenarios:
        first_line = textwrap.shorten(s.description, width=58, placeholder="...")
        print(f"{s.name:<20}  {first_line}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _list_schedulers() -> None:
    """Print a table of all registered schedulers and exit."""
    print(f"{'Name':<20}  Class")
    print("-" * 60)
    for name, cls in SCHEDULERS.items():
        print(f"{name:<20}  {cls.__module__}.{cls.__qualname__}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Run async-RL scheduling simulation scenarios.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--scenario",
        nargs="+",
        metavar="NAME",
        help="One or more scenario names to run (default: 'default').",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all registered scenarios.",
    )
    group.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List available scenarios and exit.",
    )
    group.add_argument(
        "--list-schedulers",
        action="store_true",
        dest="list_schedulers",
        help="List available schedulers and exit.",
    )
    parser.add_argument(
        "--scheduler",
        metavar="NAME",
        default="greedy-fifo",
        help="Scheduler algorithm to use (default: 'greedy-fifo').",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, resolve scenarios, and run them."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        sys.exit(0)

    if args.list_schedulers:
        _list_schedulers()
        sys.exit(0)

    try:
        scheduler = _get_scheduler(args.scheduler)
    except KeyError as exc:
        parser.error(str(exc))

    if args.run_all:
        scenarios = list_scenarios()
    elif args.scenario:
        scenarios = []
        for name in args.scenario:
            try:
                scenarios.append(get_scenario(name))
            except KeyError as exc:
                parser.error(str(exc))
    else:
        scenarios = [get_scenario("default")]

    for i, scenario in enumerate(scenarios):
        if i > 0:
            print("\n\n")
        _run_scenario(scenario, scheduler)


if __name__ == "__main__":
    main()
