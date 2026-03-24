#!/usr/bin/env python3
"""Run an async-RL scheduling simulation and print a summary.

Usage::

    uv run run_sim.py                 # default config
    uv run run_sim.py --adversarial   # high-variance config that triggers many drops
"""

from __future__ import annotations

import sys

from async_gym.simulation import (
    SimConfig,
    SimResult,
    Simulation,
    TickStats,
    constant_duration,
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
    batch_size=4,
    training_speed=20.0,
    max_staleness=3,
    seed=0,
)

ADVERSARIAL_CONFIG = SimConfig(
    n_tasks=20,
    n_trajectories=1,
    inference_capacity=4,
    judge_capacity=4,
    rollout_duration_fn=uniform_duration(1, 200),
    judge_duration_fn=constant_duration(1),
    batch_size=1,
    training_speed=1000.0,
    max_staleness=2,
    seed=42,
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
    print(f"  Batch size:           {cfg.batch_size}")
    print(
        f"  Training speed:       {cfg.training_speed}"
        f"  (duration = ceil(rollout_sample * n_trajectories * batch_size * 3 / speed))"
    )
    print(f"  Max staleness:        {cfg.max_staleness}")
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
        f"\n{'Tick':>6}  {'Inf%':>5}  {'Jdg%':>5}  {'Trn%':>5}  {'TrnDur':>6}  "
        f"{'R_disp':>6}  {'J_disp':>6}  {'R_done':>6}  {'J_done':>6}  "
        f"{'Ckpt':>4}  {'Buf':>3}  {'Stale':>5}  {'Drop':>4}  "
        f"State distribution"
    )
    print("-" * 156)

    for stats in history:
        is_periodic = stats.tick % interval == 0
        is_last = stats.tick == history[-1].tick
        if not (is_periodic or is_last):
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run the simulation with the chosen config and print results."""
    use_adversarial = "--adversarial" in sys.argv
    cfg = ADVERSARIAL_CONFIG if use_adversarial else CONFIG
    interval = 1 if use_adversarial else SNAPSHOT_INTERVAL

    _print_header(cfg)

    sim = Simulation(cfg)
    result = sim.run()

    _print_snapshots(result.history, interval)
    _print_summary(result)


if __name__ == "__main__":
    main()
