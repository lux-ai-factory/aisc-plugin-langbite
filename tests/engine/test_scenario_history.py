"""T3-T5: the run config's optional "history" field (C8, scenario side)."""
import pytest

from fakes import scenario_config
from langbite.controllers.test_scenario import TestScenario
from langbite.io_managers import json_io_manager


def test_t3_no_history_field_means_none():
    scenario = TestScenario(scenario_config(["X"]))
    assert scenario.history is None
    assert scenario.history_name is None


def test_t4_known_name_resolves_to_messages():
    name, messages = next(iter(json_io_manager.load_histories().items()))
    scenario = TestScenario(scenario_config(["X"], history=name))
    assert scenario.history_name == name
    assert scenario.history == messages


def test_t5_unknown_name_raises_at_load():
    with pytest.raises(ValueError, match="no_such_history"):
        TestScenario(scenario_config(["X"], history="no_such_history"))
