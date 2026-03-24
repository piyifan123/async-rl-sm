# Scheduling Algorithms

This document describes the dispatch-scheduling strategies available in
Async-Gym, with particular focus on the **SRPT-Aging** scheduler and the
queueing-theory reasoning behind it.

For the pluggable scheduler interface, task state machine, and simulation
mechanics, see [design.md](design.md).

---

## 1  Problem Setting

The simulation models an asynchronous off-policy RL pipeline where *N* tasks
share two finite resource pools:

| Pool       | Capacity              | Work unit      |
|------------|-----------------------|----------------|
| Inference  | `inference_capacity`  | rollout        |
| Judge      | `judge_capacity`      | judge eval     |

Each task requires `n_trajectories` rollouts followed by `n_trajectories`
judge evaluations.  Rollout and judge durations are drawn from configurable
distributions and may vary across tasks — creating a *variable job-size*
workload.

Completed tasks enter a ready buffer and are consumed in batches of
`batch_size` to trigger training.  A task whose checkpoint age (`staleness`)
exceeds `max_staleness` at consumption time is **dropped** — wasting every
resource it consumed.

The scheduler's goal is therefore:

1. **Minimise mean flow time** (ticks from admission to consumption).
2. **Maximise pool utilisation** (avoid idle slots).
3. **Minimise drops** (avoid wasting work on tasks that will be stale).

These objectives are in tension: aggressively prioritising short tasks (goal
1) can starve long tasks into staleness drops (goal 3).

---

## 2  Baseline: Greedy FIFO

`GreedyFIFOScheduler` dispatches tasks in **creation order**, giving each
task `min(pending, remaining_slots)` greedily.

| Property            | Value                                              |
|---------------------|----------------------------------------------------|
| Dispatch order      | Fixed (creation index)                             |
| Starvation          | None — every task is visited in order               |
| Responsiveness      | Low — a task with 1 remaining rollout waits behind a task with 1 000 |
| Head-of-line block  | Yes — early large tasks monopolise slots            |

FIFO is optimal for *fairness* but suboptimal for mean flow time when job
sizes are heterogeneous.

---

## 3  SRPT-Aging Scheduler

### 3.1  Intuition

**Shortest Remaining Processing Time (SRPT)** is a classic scheduling
discipline that, in a single-server M/G/1 queue, provably minimises the mean
number of jobs in the system (and therefore mean flow time by Little's law).
The key insight is that finishing short jobs first clears them from the queue
quickly, reducing the total waiting experienced by all jobs.

Pure SRPT has a well-known weakness: *starvation*.  In a stream of arrivals
where short jobs keep appearing, a long job can be deferred indefinitely.  In
this simulation the consequence is concrete — a starved task accumulates
staleness and is eventually dropped, wasting every slot it already consumed.

**Aging** is the standard remedy.  Each task accumulates a priority bonus
proportional to the time it has spent in the system.  Eventually the bonus
outweighs any size disadvantage, guaranteeing that every task will be
dispatched within a bounded time.

### 3.2  Priority Function

For each eligible task the scheduler computes a **score**:

```
score(task) = W - α · a
```

where:

| Symbol | Meaning |
|--------|---------|
| W      | Remaining work items: `pending_rollout` (rollout phase) or `pending_judge` (judge phase) |
| α      | `aging_factor` — a non-negative tuning parameter (default 1.0) |
| a      | Age in ticks: `current_tick - admission_tick` |

Tasks are sorted by **ascending score** (ties broken by creation order for
stability), then dispatched greedily with `min(pending, remaining_slots)`.

**Lower score = higher priority.**

### 3.3  Degenerate Cases

| `aging_factor` | Behaviour |
|-----------------|-----------|
| 0               | Pure SRPT — only remaining work matters |
| → ∞             | Approaches FIFO — oldest tasks first regardless of size |

The parameter interpolates continuously between these two extremes.

---

## 4  Queueing-Theory Analysis

### 4.1  SRPT Optimality (M/G/1)

Consider a single-server queue with Poisson arrivals (rate λ) and general
service-time distribution *S*.  SRPT minimises the mean number of jobs *L* in
the system among all non-preemptive and preemptive policies (Schrage, 1968).
By Little's law, L = λ · W̄, so SRPT also minimises mean sojourn time W̄.

The mean conditional sojourn time for a job of original size *x* under SRPT
in an M/G/1 queue is:

```
W_SRPT(x) = x / [(1 - ρ(x))²]
```

where ρ(x) = λ · E[S · 𝟙(S ≤ x)] is the load contributed by jobs no larger
than *x*.  Short jobs experience almost zero queueing; long jobs bear the
remaining load.

### 4.2  Starvation Under Pure SRPT

The conditional sojourn time W_SRPT(x) diverges as ρ(x) → 1.  In practical
terms: if the system is heavily loaded and short jobs arrive frequently, a
large job can wait arbitrarily long.  In this simulation, unbounded waiting
translates directly to **staleness drops** — the worst outcome since all
resources invested in the task are wasted.

### 4.3  Aging as a Bounded-Delay Guarantee

With the linear aging term the effective remaining work seen by the scheduler
is:

```
W_eff(t) = W(t) - α · a(t)
```

A task admitted at time t₀ with initial work W₀ that receives *no* service
sees its effective work decline linearly:

```
W_eff(t) = W₀ - α · (t - t₀)
```

This reaches zero at t* = t₀ + W₀/α and goes negative thereafter, meaning
the task is ranked *above* any newly arrived task regardless of size.

**Starvation bound.**  Consider two tasks competing at time *t*:

- Task A: remaining work W_A, admitted at t_A
- Task B: remaining work W_B < W_A, admitted at t_B ≥ t_A

Task A overtakes B when:

```
W_A - α(t - t_A) < W_B - α(t - t_B)
```

Solving:

```
t > t_A + (W_A - W_B) / α + (t_B - t_A)
```

In the worst case (t_B = t, a fresh arrival), the maximum additional wait for
A is:

```
Δt = (W_A - W_B) / α
```

With α = 1.0 and a work difference of 100 items, the older task is guaranteed
promotion within 100 ticks — regardless of how many small tasks arrive in
the interim.

### 4.4  Relationship to Multi-Level Feedback Queues

The aging mechanism is conceptually equivalent to a **Multi-Level Feedback
Queue (MLFQ)** where:

- New tasks start in a high-priority queue (low remaining work relative to
  age).
- Tasks that have not been served are implicitly "promoted" as their age
  increases.
- The single linear score replaces discrete priority levels with a continuous
  spectrum.

Unlike a classical MLFQ, there are no explicit queue levels or time-quantum
resets.  The linear score provides a simpler, parameter-efficient
approximation.

### 4.5  Flow-Time Improvement Over FIFO

Under a heavy-tailed job-size distribution (common in RL where problem
difficulty varies widely), SRPT-Aging delivers significantly lower mean flow
time than FIFO.  Intuitively:

- Short tasks (easy problems) complete in a few ticks instead of waiting
  behind large tasks.
- The aging guarantee ensures long tasks still complete before their staleness
  deadline.

The magnitude of improvement depends on the coefficient of variation (CV) of
the job-size distribution.  For CV ≫ 1 (heavy tail), theory predicts
SRPT-family policies can reduce mean sojourn time by a factor proportional to
the tail index.  For CV ≈ 0 (constant durations), SRPT degenerates to FIFO
since all jobs have similar remaining work.

---

## 5  Implementation Notes

### 5.1  Stateful Scheduler

`SRPTAgingScheduler` maintains an internal dictionary
`_admission_ticks: dict[str, int]` that maps each task's ID to the tick when
it was admitted.  This is recorded in `should_admit` and read during dispatch.

### 5.2  Tie-Breaking

When two tasks have the same score, the scheduler preserves their original
position in the task list (creation order).  This is achieved by using
`(score, original_index)` as the sort key, ensuring a **stable** sort.

### 5.3  Greedy Allocation

After sorting by priority, allocation follows the same greedy pattern as the
baseline: each task receives `min(pending_work, remaining_slots)`.  This is
optimal given the sorted order — there is no benefit to under-allocating a
higher-priority task to "save" slots for a lower-priority one, because the
higher-priority task is expected to finish sooner and free slots earlier.

### 5.4  Admission Policy

Admission is intentionally kept identical to the baseline (`active_count <
pipeline_cap`).  Admission controls the *number* of tasks in flight; dispatch
controls the *allocation of resources* among them.  Decoupling these concerns
allows the scheduling strategy to be evaluated independently of pipeline
depth tuning.

---

## 6  Parameter Tuning Guidance

### 6.1  Choosing `aging_factor`

The key relationship is:

```
max_wait_before_promotion ≈ ΔW / α
```

where ΔW is the largest work-size difference between any two concurrent
tasks (roughly `max(n_trajectories) - min(n_trajectories)` among active
tasks, measured in the pending-work dimension being scheduled).

| Scenario | Guidance |
|----------|----------|
| Homogeneous `n_trajectories` | α matters little — all tasks have similar remaining work.  Default 1.0 is fine. |
| Heterogeneous sizes, tight `max_staleness` | Increase α so that promotion happens well before the staleness deadline: α > ΔW / (staleness_budget_in_ticks). |
| Heterogeneous sizes, relaxed `max_staleness` | Default α = 1.0 often suffices.  Decrease α toward 0 to lean more toward pure SRPT for lower mean flow time. |

### 6.2  Rules of Thumb

1. **Start with α = 1.0.**  This gives a promotion horizon equal to the work
   difference in ticks — a reasonable default.
2. **If drops increase**, raise α to promote old tasks faster.
3. **If mean flow time increases**, lower α to lean toward SRPT.
4. **If all tasks have the same `n_trajectories`**, the scheduler is
   effectively FIFO with a small transient SRPT effect from partially
   completed work — α has minimal impact.

### 6.3  Interaction with `max_staleness` and `pipeline_cap`

The pipeline cap limits how many tasks are active simultaneously.  With a
small pipeline cap, fewer tasks compete and the scheduling order matters
less.  The SRPT-Aging advantage is most pronounced when:

- `pipeline_cap` is large (many competing tasks).
- Job sizes are heterogeneous.
- `max_staleness` is tight enough that starvation leads to drops.

---

## 7  Summary

| Scheduler      | Dispatch order                | Starvation-free | Mean flow time |
|----------------|-------------------------------|-----------------|----------------|
| Greedy FIFO    | Creation order                | Yes             | Baseline       |
| SRPT (α = 0)   | Shortest remaining first     | No              | Optimal (M/G/1)|
| SRPT-Aging     | Shortest remaining + age boost| Yes (bounded)   | Near-optimal   |

SRPT-Aging combines the flow-time benefits of SRPT with the starvation
guarantees needed to avoid wasteful drops in the async-RL pipeline.  The
single `aging_factor` parameter provides a simple, interpretable knob for
trading off responsiveness against fairness.
