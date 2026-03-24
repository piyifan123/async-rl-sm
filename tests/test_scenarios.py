"""Tests for the scenario registry."""

from __future__ import annotations

import pytest

from async_gym.scenarios import (
    SCENARIOS,
    Scenario,
    get_scenario,
    list_scenarios,
)
from async_gym.simulation import SimConfig, Simulation

# ------------------------------------------------------------------
# Registry contents
# ------------------------------------------------------------------


class TestRegistryContents:
    def test_at_least_four_scenarios_registered(self) -> None:
        assert len(SCENARIOS) >= 4

    def test_expected_names_present(self) -> None:
        expected = {"default", "adversarial", "small-constant", "high-throughput"}
        assert expected.issubset(SCENARIOS.keys())

    def test_all_entries_are_scenario_instances(self) -> None:
        for scenario in SCENARIOS.values():
            assert isinstance(scenario, Scenario)

    def test_all_configs_are_sim_configs(self) -> None:
        for scenario in SCENARIOS.values():
            assert isinstance(scenario.config, SimConfig)

    def test_names_match_keys(self) -> None:
        for key, scenario in SCENARIOS.items():
            assert key == scenario.name

    def test_all_have_nonempty_descriptions(self) -> None:
        for scenario in SCENARIOS.values():
            assert scenario.description.strip(), f"{scenario.name} has empty description"

    def test_snapshot_intervals_positive(self) -> None:
        for scenario in SCENARIOS.values():
            assert scenario.snapshot_interval >= 1, (
                f"{scenario.name} has non-positive snapshot_interval"
            )


# ------------------------------------------------------------------
# Lookup helpers
# ------------------------------------------------------------------


class TestGetScenario:
    def test_returns_correct_scenario(self) -> None:
        s = get_scenario("default")
        assert s.name == "default"
        assert s is SCENARIOS["default"]

    def test_all_registered_names_resolve(self) -> None:
        for name in SCENARIOS:
            s = get_scenario(name)
            assert s.name == name

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="unknown scenario"):
            get_scenario("nonexistent-scenario")

    def test_error_message_lists_available(self) -> None:
        with pytest.raises(KeyError, match="available:"):
            get_scenario("nope")


class TestListScenarios:
    def test_returns_all_registered(self) -> None:
        result = list_scenarios()
        assert len(result) == len(SCENARIOS)

    def test_returns_list_of_scenarios(self) -> None:
        for item in list_scenarios():
            assert isinstance(item, Scenario)

    def test_preserves_registration_order(self) -> None:
        names = [s.name for s in list_scenarios()]
        assert names == list(SCENARIOS.keys())


# ------------------------------------------------------------------
# Smoke-run each scenario
# ------------------------------------------------------------------


class TestScenarioSmokeRun:
    """Run each registered scenario for a small number of ticks to verify
    it produces a valid result without errors."""

    @pytest.fixture(params=list(SCENARIOS.keys()))
    def scenario(self, request: pytest.FixtureRequest) -> Scenario:
        """Parametrize over all registered scenarios."""
        return get_scenario(request.param)

    def test_runs_without_error(self, scenario: Scenario) -> None:
        sim = Simulation(scenario.config)
        result = sim.run()
        assert result.ticks_elapsed > 0
        assert result.tasks_completed + result.tasks_dropped == scenario.config.n_tasks

    def test_history_length_matches_ticks(self, scenario: Scenario) -> None:
        result = Simulation(scenario.config).run()
        assert len(result.history) == result.ticks_elapsed
