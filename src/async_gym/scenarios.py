"""Centralized registry of named simulation scenarios.

Each :class:`Scenario` bundles a :class:`SimConfig` with a human-readable
name, description, and display settings so that ``run_sim.py`` (and users)
can refer to scenarios by short identifiers instead of constructing configs
by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from async_gym.simulation import (
    SimConfig,
    bimodal_duration,
    constant_duration,
    uniform_duration,
)

__all__ = [
    "Scenario",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
]


@dataclass(frozen=True)
class Scenario:
    """A named, self-describing simulation scenario.

    Args:
        name: Short kebab-case identifier (e.g. ``"default"``).
        description: 1-3 sentence explanation of what the scenario demonstrates.
        config: The simulation parameters.
        snapshot_interval: Tick interval for printing periodic snapshots.
    """

    name: str
    description: str
    config: SimConfig
    snapshot_interval: int = 10


# ------------------------------------------------------------------
# Built-in scenarios
# ------------------------------------------------------------------

SCENARIOS: dict[str, Scenario] = {}


def _register(scenario: Scenario) -> Scenario:
    """Add *scenario* to the global registry, keyed by its name.

    Args:
        scenario: The scenario to register.

    Returns:
        The same scenario (for convenience at definition site).

    Raises:
        ValueError: If a scenario with the same name is already registered.
    """
    if scenario.name in SCENARIOS:
        raise ValueError(f"duplicate scenario name: {scenario.name!r}")
    SCENARIOS[scenario.name] = scenario
    return scenario


_register(
    Scenario(
        name="default",
        description=(
            "Balanced workload with moderate parallelism. "
            "Demonstrates steady-state pipeline utilization across "
            "inference, judging, and training phases."
        ),
        config=SimConfig(
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
        ),
        snapshot_interval=10,
    )
)

_register(
    Scenario(
        name="adversarial",
        description=(
            "Extreme rollout variance (1-200 ticks) with tight staleness "
            "bound and fast training. Demonstrates task drops caused by "
            "slow tasks being lapped by checkpoint advances."
        ),
        config=SimConfig(
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
        ),
        snapshot_interval=25,
    )
)

_register(
    Scenario(
        name="small-constant",
        description=(
            "Minimal scenario with constant durations and few tasks. "
            "Useful for smoke testing, debugging, and understanding "
            "the basic pipeline flow."
        ),
        config=SimConfig(
            n_tasks=4,
            n_trajectories=2,
            inference_capacity=4,
            judge_capacity=2,
            rollout_duration_fn=constant_duration(2),
            judge_duration_fn=constant_duration(1),
            batch_size=2,
            training_speed=10.0,
            max_staleness=3,
            seed=7,
        ),
        snapshot_interval=1,
    )
)

_register(
    Scenario(
        name="high-throughput",
        description=(
            "High parallelism with large batch size, many inference "
            "replicas, and fast training. Demonstrates batched training "
            "throughput with many concurrent rollouts."
        ),
        config=SimConfig(
            n_tasks=40,
            n_trajectories=4,
            inference_capacity=16,
            judge_capacity=8,
            rollout_duration_fn=uniform_duration(1, 5),
            judge_duration_fn=uniform_duration(1, 3),
            batch_size=8,
            training_speed=50.0,
            max_staleness=4,
            seed=123,
        ),
        snapshot_interval=5,
    )
)

# ------------------------------------------------------------------
# SRPT-favorable scenarios
#
# These scenarios are designed so that SRPTAgingScheduler demonstrably
# outperforms GreedyFIFOScheduler.  The common pattern is a judge-capacity
# bottleneck combined with high-variance rollout durations: many tasks
# enter the judge queue simultaneously with heterogeneous pending_judge
# counts, and SRPT clears nearly-done tasks first — reducing mean flow
# time, raising utilisation, and (in some seeds) avoiding staleness drops.
# ------------------------------------------------------------------

_register(
    Scenario(
        name="srpt-bimodal",
        description=(
            "Bimodal rollout durations (80% short, 20% long) with a tight "
            "judge bottleneck.  SRPT-Aging finishes ~40% faster than FIFO, "
            "avoids 2 staleness drops, and nearly doubles resource utilisation "
            "by clearing nearly-done tasks first."
        ),
        config=SimConfig(
            n_tasks=24,
            n_trajectories=8,
            inference_capacity=16,
            judge_capacity=3,
            rollout_duration_fn=bimodal_duration(1, 50, p_short=0.8),
            judge_duration_fn=constant_duration(2),
            batch_size=4,
            training_speed=30.0,
            max_staleness=3,
            seed=40,
        ),
        snapshot_interval=10,
    )
)

_register(
    Scenario(
        name="srpt-heavy-tail",
        description=(
            "Heavy-tailed bimodal rollouts (85% 1-tick, 15% 60-tick) with "
            "32 tasks competing for scarce judge capacity.  SRPT-Aging "
            "finishes ~35% faster and drops 1 fewer task than FIFO by "
            "prioritising tasks closest to READY."
        ),
        config=SimConfig(
            n_tasks=32,
            n_trajectories=8,
            inference_capacity=16,
            judge_capacity=3,
            rollout_duration_fn=bimodal_duration(1, 60, p_short=0.85),
            judge_duration_fn=constant_duration(2),
            batch_size=4,
            training_speed=30.0,
            max_staleness=3,
            seed=62,
        ),
        snapshot_interval=15,
    )
)

_register(
    Scenario(
        name="srpt-contention",
        description=(
            "Bimodal rollouts with a larger but slower judge pool "
            "(4 slots x 3-tick duration).  No tasks are dropped under "
            "either scheduler, isolating the pure throughput advantage: "
            "SRPT-Aging finishes ~27% faster with higher utilisation."
        ),
        config=SimConfig(
            n_tasks=24,
            n_trajectories=8,
            inference_capacity=16,
            judge_capacity=4,
            rollout_duration_fn=bimodal_duration(1, 50, p_short=0.8),
            judge_duration_fn=constant_duration(3),
            batch_size=4,
            training_speed=30.0,
            max_staleness=3,
            seed=163,
        ),
        snapshot_interval=15,
    )
)


# ------------------------------------------------------------------
# Public lookup helpers
# ------------------------------------------------------------------


def get_scenario(name: str) -> Scenario:
    """Look up a registered scenario by name.

    Args:
        name: The scenario identifier (case-sensitive, kebab-case).

    Returns:
        The matching :class:`Scenario`.

    Raises:
        KeyError: If no scenario with *name* exists.

    Examples:
        >>> s = get_scenario("default")
        >>> s.name
        'default'
    """
    try:
        return SCENARIOS[name]
    except KeyError:
        available = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {name!r}; available: {available}") from None


def list_scenarios() -> list[Scenario]:
    """Return all registered scenarios in registration order.

    Returns:
        A list of :class:`Scenario` instances.

    Examples:
        >>> names = [s.name for s in list_scenarios()]
        >>> "default" in names
        True
    """
    return list(SCENARIOS.values())
