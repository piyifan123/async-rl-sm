"""Tests for bimodal_duration and SRPT-vs-FIFO scheduler comparisons.

The SRPT-favorable scenarios use fixed seeds that produce deterministic
outcomes where SRPTAgingScheduler outperforms GreedyFIFOScheduler on
ticks_elapsed, tasks_dropped, or both.
"""

from __future__ import annotations

import numpy as np
import pytest

from async_gym.scenarios import get_scenario
from async_gym.scheduler import GreedyFIFOScheduler, SRPTAgingScheduler
from async_gym.simulation import Simulation, bimodal_duration

# ------------------------------------------------------------------
# bimodal_duration unit tests
# ------------------------------------------------------------------


class TestBimodalDuration:
    def test_only_produces_short_or_long(self) -> None:
        fn = bimodal_duration(3, 50, p_short=0.8)
        samples = fn(5000, np.random.default_rng(0))
        assert set(samples) == {3, 50}

    def test_respects_p_short_approximately(self) -> None:
        fn = bimodal_duration(2, 40, p_short=0.8)
        samples = fn(10_000, np.random.default_rng(42))
        short_frac = sum(1 for s in samples if s == 2) / len(samples)
        assert 0.75 < short_frac < 0.85

    def test_description_attribute(self) -> None:
        fn = bimodal_duration(1, 60, p_short=0.85)
        assert fn.description == "bimodal(1, 60, p=0.85)"  # type: ignore[attr-defined]

    def test_deterministic_with_same_seed(self) -> None:
        fn = bimodal_duration(5, 100, p_short=0.7)
        a = fn(100, np.random.default_rng(7))
        b = fn(100, np.random.default_rng(7))
        assert a == b

    def test_rejects_short_below_one(self) -> None:
        with pytest.raises(ValueError, match="short must be >= 1"):
            bimodal_duration(0, 10)

    def test_rejects_long_not_greater_than_short(self) -> None:
        with pytest.raises(ValueError, match="long must be > short"):
            bimodal_duration(10, 10)

    def test_rejects_p_short_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="p_short must be in"):
            bimodal_duration(1, 10, p_short=0.0)
        with pytest.raises(ValueError, match="p_short must be in"):
            bimodal_duration(1, 10, p_short=1.0)


# ------------------------------------------------------------------
# SRPT-vs-FIFO comparison tests
# ------------------------------------------------------------------


class TestSRPTBimodal:
    """SRPT-Aging beats FIFO on the ``srpt-bimodal`` scenario."""

    @pytest.fixture(scope="class")
    def results(self) -> tuple:
        """Run the scenario once with each scheduler."""
        cfg = get_scenario("srpt-bimodal").config
        fifo = Simulation(cfg, GreedyFIFOScheduler()).run()
        srpt = Simulation(cfg, SRPTAgingScheduler()).run()
        return fifo, srpt

    def test_srpt_finishes_faster(self, results: tuple) -> None:
        fifo, srpt = results
        assert srpt.ticks_elapsed < fifo.ticks_elapsed

    def test_srpt_no_more_drops(self, results: tuple) -> None:
        fifo, srpt = results
        assert srpt.tasks_dropped <= fifo.tasks_dropped

    def test_srpt_higher_judge_utilisation(self, results: tuple) -> None:
        fifo, srpt = results
        avg_fifo = sum(s.judge_utilization for s in fifo.history) / len(fifo.history)
        avg_srpt = sum(s.judge_utilization for s in srpt.history) / len(srpt.history)
        assert avg_srpt > avg_fifo


class TestSRPTHeavyTail:
    """SRPT-Aging beats FIFO on the ``srpt-heavy-tail`` scenario.

    With staleness-aware admission, SRPT-Aging trades a small amount of
    wall-clock time for zero drops.  FIFO appears "faster" only because
    it gives up on tasks that go stale.
    """

    @pytest.fixture(scope="class")
    def results(self) -> tuple:
        """Run the scenario once with each scheduler."""
        cfg = get_scenario("srpt-heavy-tail").config
        fifo = Simulation(cfg, GreedyFIFOScheduler()).run()
        srpt = Simulation(cfg, SRPTAgingScheduler()).run()
        return fifo, srpt

    def test_srpt_completes_more_tasks(self, results: tuple) -> None:
        fifo, srpt = results
        assert srpt.tasks_completed >= fifo.tasks_completed

    def test_srpt_fewer_drops(self, results: tuple) -> None:
        fifo, srpt = results
        assert srpt.tasks_dropped <= fifo.tasks_dropped


class TestSRPTContention:
    """SRPT-Aging beats FIFO on the ``srpt-contention`` scenario (pure throughput)."""

    @pytest.fixture(scope="class")
    def results(self) -> tuple:
        """Run the scenario once with each scheduler."""
        cfg = get_scenario("srpt-contention").config
        fifo = Simulation(cfg, GreedyFIFOScheduler()).run()
        srpt = Simulation(cfg, SRPTAgingScheduler()).run()
        return fifo, srpt

    def test_srpt_finishes_faster(self, results: tuple) -> None:
        fifo, srpt = results
        assert srpt.ticks_elapsed < fifo.ticks_elapsed

    def test_no_drops_either_scheduler(self, results: tuple) -> None:
        fifo, srpt = results
        assert fifo.tasks_dropped == 0
        assert srpt.tasks_dropped == 0

    def test_srpt_higher_inference_utilisation(self, results: tuple) -> None:
        fifo, srpt = results
        avg_fifo = sum(s.inference_utilization for s in fifo.history) / len(fifo.history)
        avg_srpt = sum(s.inference_utilization for s in srpt.history) / len(srpt.history)
        assert avg_srpt > avg_fifo


# ------------------------------------------------------------------
# Staleness-aware admission integration tests
# ------------------------------------------------------------------


class TestStalenessAdmissionAdversarial:
    """The staleness-aware gate eliminates drops on the adversarial scenario.

    Without the gate (``admit_headroom=None``), SRPT-Aging on this scenario
    still drops tasks because fast tasks cycle through the pipeline and
    advance the checkpoint while slow tasks are stuck.  With the default
    headroom, admission is throttled to protect slow tasks.
    """

    @pytest.fixture(scope="class")
    def results(self) -> tuple:
        cfg = get_scenario("adversarial").config
        no_gate = Simulation(cfg, SRPTAgingScheduler(admit_headroom=None)).run()
        with_gate = Simulation(cfg, SRPTAgingScheduler(admit_headroom=1)).run()
        return no_gate, with_gate

    def test_gate_reduces_drops(self, results: tuple) -> None:
        no_gate, with_gate = results
        assert with_gate.tasks_dropped < no_gate.tasks_dropped

    def test_gate_completes_more_tasks(self, results: tuple) -> None:
        no_gate, with_gate = results
        assert with_gate.tasks_completed >= no_gate.tasks_completed

    def test_all_tasks_accounted_for(self, results: tuple) -> None:
        cfg = get_scenario("adversarial").config
        for result in results:
            assert result.tasks_completed + result.tasks_dropped == cfg.n_tasks


class TestStalenessAdmissionHeavyTail:
    """The staleness-aware gate eliminates drops on the heavy-tail scenario."""

    @pytest.fixture(scope="class")
    def results(self) -> tuple:
        cfg = get_scenario("srpt-heavy-tail").config
        no_gate = Simulation(cfg, SRPTAgingScheduler(admit_headroom=None)).run()
        with_gate = Simulation(cfg, SRPTAgingScheduler(admit_headroom=1)).run()
        return no_gate, with_gate

    def test_gate_eliminates_drops(self, results: tuple) -> None:
        _no_gate, with_gate = results
        assert with_gate.tasks_dropped == 0

    def test_no_gate_has_drops(self, results: tuple) -> None:
        no_gate, _with_gate = results
        assert no_gate.tasks_dropped > 0

    def test_gate_completes_all_tasks(self, results: tuple) -> None:
        cfg = get_scenario("srpt-heavy-tail").config
        _no_gate, with_gate = results
        assert with_gate.tasks_completed == cfg.n_tasks


class TestStalenessAdmissionBimodal:
    """The staleness-aware gate on the bimodal scenario (already 0 drops)."""

    @pytest.fixture(scope="class")
    def results(self) -> tuple:
        cfg = get_scenario("srpt-bimodal").config
        no_gate = Simulation(cfg, SRPTAgingScheduler(admit_headroom=None)).run()
        with_gate = Simulation(cfg, SRPTAgingScheduler(admit_headroom=1)).run()
        return no_gate, with_gate

    def test_gate_no_worse_drops(self, results: tuple) -> None:
        no_gate, with_gate = results
        assert with_gate.tasks_dropped <= no_gate.tasks_dropped

    def test_all_tasks_accounted_for(self, results: tuple) -> None:
        cfg = get_scenario("srpt-bimodal").config
        for result in results:
            assert result.tasks_completed + result.tasks_dropped == cfg.n_tasks
