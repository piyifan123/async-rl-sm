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
uv run run_sim.py
```

This runs a default scenario (20 tasks, 8 trajectories each, 8 inference / 4
judge replicas, uniform rollout durations in [2, 10] ticks, uniform judge
durations in [1, 4] ticks) and prints periodic snapshots with a final summary:

```
Async-RL Scheduling Simulation
  Tasks:                20
  Trajectories/task:    8
  Inference capacity:   8
  Judge capacity:       4
  Max ticks:            100000
  Seed:                 0

  Tick   Inf%   Jdg%  R_disp  J_disp  R_done  J_done  State distribution
     0  100.0%   0.0%       8       0       0       0  pending=19  rolling_out=1  partial=0  judging=0  ready=0  consumed=0
    10  100.0%  75.0%       1       1       2       1  pending=17  rolling_out=0  partial=2  judging=0  ready=0  consumed=1
    ...
   123   0.0%  25.0%       0       0       0       1  pending=0  rolling_out=0  partial=0  judging=0  ready=0  consumed=20

Summary
  Ticks elapsed:        124
  Tasks completed:      20
  Avg inference util:   94.9%
  Avg judge util:       87.9%
```

To customise the simulation, edit the `CONFIG` object in `run_sim.py` or build
your own script using the library directly:

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
│   └── simulation.py        # SimConfig, Simulation runner, duration helpers
├── tests/
│   ├── test_task.py
│   ├── test_replica_pool.py
│   └── test_simulation.py
├── docs/
│   └── design.md            # Architecture and design decisions
├── run_sim.py               # Entry-point script
└── pyproject.toml
```

## Design

See [docs/design.md](docs/design.md) for the full design document covering the
task state machine, replica pools, and simulation runner architecture.
