# Async-Gym

Discrete-event simulation of an asynchronous, off-policy RL scheduling pipeline.

Tasks move through a five-stage trajectory pipeline (pending rollout, rolling
out, pending judge, judging, judged) while competing for shared inference and
judge replica pools. A greedy scheduler dispatches work each tick, respecting
pool capacity constraints.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

## Running the simulation

```bash
uv run run_sim.py                                # run the "default" scenario
uv run run_sim.py --scenario adversarial         # run one scenario by name
uv run run_sim.py --scenario default adversarial # run several by name
uv run run_sim.py --all                          # run all registered scenarios
uv run run_sim.py --list                         # list available scenarios and exit
```

### Available scenarios

| Name | Description |
|------|-------------|
| `default` | Balanced workload with moderate parallelism. Demonstrates steady-state pipeline utilization. |
| `adversarial` | Extreme rollout variance (1-200 ticks) with tight staleness. Demonstrates task drops from checkpoint lapping. |
| `small-constant` | Minimal scenario with constant durations and few tasks. Useful for smoke testing and debugging. |
| `high-throughput` | High parallelism with large batch size and fast training. Demonstrates batched training throughput. |

Each scenario prints its name, description, configuration, periodic tick
snapshots, and a final summary.

### Using the library directly

You can also build custom configs without registering them as scenarios:

```python
from async_gym import SimConfig, Simulation, constant_duration, uniform_duration

config = SimConfig(
    n_tasks=10,
    n_trajectories=4,
    inference_capacity=4,
    judge_capacity=2,
    rollout_duration_fn=uniform_duration(2, 8),
    judge_duration_fn=constant_duration(2),
    seed=42,
)

result = Simulation(config).run()
print(f"Completed {result.tasks_completed} tasks in {result.ticks_elapsed} ticks")
```

## Running tests

```bash
uv run pytest
```

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
```

## Project structure

```
async-rl-state/
├── src/async_gym/
│   ├── __init__.py          # Public API exports
│   ├── task.py              # Task state machine (per-task trajectory pipeline)
│   ├── replica_pool.py      # ReplicaPool (shared capacity constraint)
│   ├── simulation.py        # SimConfig, Simulation runner, duration helpers
│   └── scenarios.py         # Named scenario registry (Scenario, get_scenario, list_scenarios)
├── tests/
│   ├── test_task.py
│   ├── test_replica_pool.py
│   ├── test_simulation.py
│   └── test_scenarios.py
├── docs/
│   └── design.md            # Architecture and design decisions
├── run_sim.py               # Entry-point script (multi-scenario CLI)
└── pyproject.toml
```

## Design

See [docs/design.md](docs/design.md) for the full design document covering the
task state machine, replica pools, and simulation runner architecture.
