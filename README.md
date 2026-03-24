# Async-Gym

Discrete-event simulation of an asynchronous, off-policy RL scheduling pipeline.

Tasks move through a five-stage trajectory pipeline (pending rollout, rolling
out, pending judge, judging, judged) while competing for shared inference and
judge replica pools. A pluggable scheduler dispatches work each tick, respecting
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
uv run run_sim.py                                        # default scenario, greedy-fifo scheduler
uv run run_sim.py --scenario adversarial                 # run one scenario by name
uv run run_sim.py --scenario default adversarial         # run several scenarios
uv run run_sim.py --all                                  # run all registered scenarios
uv run run_sim.py --scheduler greedy-fifo                # specify scheduler algorithm
uv run run_sim.py --scenario default --scheduler greedy-fifo  # combine both
uv run run_sim.py --list                                 # list available scenarios
uv run run_sim.py --list-schedulers                      # list available schedulers
```

### Available scenarios (benchmarks)

Scenarios define the **environment** — task count, capacities, duration
distributions, staleness bounds.  Use `--list` to see all registered scenarios.

| Name | Description |
|------|-------------|
| `default` | Balanced workload with moderate parallelism. Demonstrates steady-state pipeline utilization. |
| `adversarial` | Extreme rollout variance (1-200 ticks) with tight staleness. Demonstrates task drops from checkpoint lapping. |
| `small-constant` | Minimal scenario with constant durations and few tasks. Useful for smoke testing and debugging. |
| `high-throughput` | High parallelism with large batch size and fast training. Demonstrates batched training throughput. |

### Available schedulers

Schedulers define the **policy** — how resources are allocated to tasks each
tick.  Use `--list-schedulers` to see all registered schedulers.

| Name | Strategy |
|------|----------|
| `greedy-fifo` | Iterate tasks in creation order; each task greedily takes `min(pending, available)` slots. Pipeline-cap admission. |

### Comparing schedulers

To compare scheduling algorithms, run the same scenario with different
schedulers:

```bash
uv run run_sim.py --scenario adversarial --scheduler greedy-fifo
```

Or compare programmatically:

```python
from async_gym import SimConfig, Simulation, GreedyFIFOScheduler, get_scenario

config = get_scenario("default").config
for scheduler in [GreedyFIFOScheduler()]:
    result = Simulation(config, scheduler=scheduler).run()
    print(f"{scheduler.name}: {result.ticks_elapsed} ticks, "
          f"{result.tasks_dropped} dropped")
```

### Using the library directly

Build custom configs and schedulers without registering them:

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

### Implementing a custom scheduler

Subclass `Scheduler` (or `GreedyFIFOScheduler` to override a single hook):

```python
from async_gym import GreedyFIFOScheduler, Scheduler, SchedulerView, DispatchAction
from async_gym.task import Task

class MyScheduler(GreedyFIFOScheduler):
    @property
    def name(self) -> str:
        return "my-scheduler"

    def should_admit(self, task: Task, active_count: int, view: SchedulerView) -> bool:
        # Back-pressure: don't admit while training is running
        if view.training_in_flight:
            return False
        return super().should_admit(task, active_count, view)
```

See [docs/design.md](docs/design.md) §6 for the full scheduler interface
contract and more examples.

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
│   ├── scheduler.py         # Scheduler ABC, GreedyFIFOScheduler, DispatchAction
│   ├── simulation.py        # SimConfig, Simulation runner, duration helpers
│   └── scenarios.py         # Named scenario registry (Scenario, get_scenario, list_scenarios)
├── tests/
│   ├── test_task.py
│   ├── test_replica_pool.py
│   ├── test_simulation.py
│   ├── test_scheduler.py
│   └── test_scenarios.py
├── docs/
│   └── design.md            # Architecture and design decisions
├── run_sim.py               # Entry-point script (multi-scenario, multi-scheduler CLI)
└── pyproject.toml
```

## Design

See [docs/design.md](docs/design.md) for the full design document covering the
task state machine, replica pools, simulation runner, and pluggable scheduler
architecture.
