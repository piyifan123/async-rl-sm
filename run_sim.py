#!/usr/bin/env python3
"""Run a default async-RL scheduling simulation and print a summary.

Usage::

    uv run run_sim.py
"""

from __future__ import annotations

from async_gym.simulation import (
    SimConfig,
    SimResult,
    Simulation,
    TickStats,
    describe_duration_fn,
    uniform_duration,
)
from async_gym.task import TaskState

# ------------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------------

CONFIG = SimConfig(
    n_tasks=20,
    n_trajectories=8,
    inference_capacity=8,
    judge_capacity=4,
    rollout_duration_fn=uniform_duration(2, 10),
    judge_duration_fn=uniform_duration(1, 4),
    seed=0,
)

SNAPSHOT_INTERVAL = 10


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
    }
    parts = []
    for state in _STATE_ORDER:
        count = stats.tasks_by_state.get(state, 0)
        parts.append(f"{labels[state.name]}={count}")
    return "  ".join(parts)


def _print_header(cfg: SimConfig) -> None:
    """Print the simulation configuration header.

    Args:
        cfg: The simulation config to summarise.
    """
    print("=" * 120)
    print("Async-RL Scheduling Simulation")
    print("=" * 120)
    print(f"  Tasks:                {cfg.n_tasks}")
    print(f"  Trajectories/task:    {cfg.n_trajectories}")
    print(f"  Inference capacity:   {cfg.inference_capacity}")
    print(f"  Judge capacity:       {cfg.judge_capacity}")
    print(f"  Rollout duration:     {describe_duration_fn(cfg.rollout_duration_fn)}")
    print(f"  Judge duration:       {describe_duration_fn(cfg.judge_duration_fn)}")
    print(f"  Max ticks:            {cfg.max_ticks}")
    print(f"  Seed:                 {cfg.seed}")
    print("-" * 120)


def _print_snapshots(history: list[TickStats], interval: int) -> None:
    """Print periodic tick snapshots.

    Args:
        history: Full tick history from the simulation.
        interval: Print every *interval*-th tick plus the final tick.
    """
    print(
        f"\n{'Tick':>6}  {'Inf%':>5}  {'Jdg%':>5}  {'R_disp':>6}  "
        f"{'J_disp':>6}  {'R_done':>6}  {'J_done':>6}  State distribution"
    )
    print("-" * 120)

    for stats in history:
        is_periodic = stats.tick % interval == 0
        is_last = stats.tick == history[-1].tick
        if not (is_periodic or is_last):
            continue

        print(
            f"{stats.tick:>6}  "
            f"{stats.inference_utilization:>5.1%}  "
            f"{stats.judge_utilization:>5.1%}  "
            f"{stats.rollouts_dispatched:>6}  "
            f"{stats.judges_dispatched:>6}  "
            f"{stats.rollouts_completed:>6}  "
            f"{stats.judges_completed:>6}  "
            f"{_format_state_distribution(stats)}"
        )


def _print_summary(result: SimResult) -> None:
    """Print the final summary.

    Args:
        result: The simulation result.
    """
    avg_inf = (
        sum(s.inference_utilization for s in result.history) / len(result.history)
        if result.history
        else 0.0
    )
    avg_jdg = (
        sum(s.judge_utilization for s in result.history) / len(result.history)
        if result.history
        else 0.0
    )

    print("\n" + "=" * 120)
    print("Summary")
    print("=" * 120)
    print(f"  Ticks elapsed:        {result.ticks_elapsed}")
    print(f"  Tasks completed:      {result.tasks_completed}")
    print(f"  Avg inference util:   {avg_inf:.1%}")
    print(f"  Avg judge util:       {avg_jdg:.1%}")
    print("=" * 120)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run the simulation with the default config and print results."""
    _print_header(CONFIG)

    sim = Simulation(CONFIG)
    result = sim.run()

    _print_snapshots(result.history, SNAPSHOT_INTERVAL)
    _print_summary(result)


if __name__ == "__main__":
    main()
