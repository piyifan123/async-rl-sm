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
