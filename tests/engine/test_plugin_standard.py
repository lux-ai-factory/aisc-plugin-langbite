"""The LangBiTe plugin follows the AISC standard (aisc docs/superpowers/target-standard-2026-10-04/01-plan.md):
the model under test is the evaluation's target, reached through its Manage endpoint only; the judge
(LLMEval) is a tool model with its own fields on the execution page; a conversation history can be
chosen. Run with the stack's plugin interface:

    uv run --with pytest --with-editable <aisc>/shared/plugin-interface python -m pytest -q tests
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aisc_plugin_interface.system_under_test import SYSTEM_INPUT, target_access_of  # noqa: E402
from aisc_plugin_langbite.models import ConfigFormSchema  # noqa: E402
from aisc_plugin_langbite.plugin import LangBiteEvaluationPlugin  # noqa: E402

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
BASE = f"http://platform:8000/internal/facade/{PID}/mcas-chat/openai/v1"
ISSUED = {"key": "aisc-run-key", "connection": "mcas-chat", "capabilities": ["text", "history"],
          "target": {"key": "system", "label": "MCAS"},
          "endpoints": {"openai": {"base_url": BASE, "model": "mcas-chat"}}}
REQUIREMENT = {"name": "REQ1", "rationale": "test", "tolerance": 0.9, "delta": 0.02, "concern": "Racism",
               "markup": "SKIN_COLOR", "communities": [{"language": "en_us", "entries": ["Black", "White"]}],
               "inputs": ["constrained"], "reflections": ["utopian"]}


def form(**over):
    data = {"nTemplates": 60, "nRetries": 1, "temperature": 1.0, "tokens": 60, "requirements": [REQUIREMENT]}
    data.update(over)
    return data


def test_it_declares_that_it_reaches_its_target_through_its_endpoint():
    assert target_access_of(LangBiteEvaluationPlugin) == "endpoint"
    assert SYSTEM_INPUT in {d.name for d in LangBiteEvaluationPlugin().input_definitions}


def test_the_form_has_no_target_model_or_key_and_gives_the_judge_its_own():
    fields = ConfigFormSchema.model_fields
    assert "aiModels" not in fields and "model_credential" not in fields
    assert {"judge_model", "judge_api_key", "history"} <= set(fields)
    assert fields["useLLMEval"].default is False                  # the judge runs only when asked for
    assert fields["nRetries"].title == "Number of retries"


def test_the_judge_model_list_holds_openai_models_only():
    schema = ConfigFormSchema.model_json_schema()
    judge = schema["$defs"][schema["properties"]["judge_model"]["$ref"].split("/")[-1]]["enum"]
    assert "OpenAIGPT4oMini" in judge and not [k for k in judge if not k.startswith("OpenAI")]


def test_the_history_list_holds_none_and_every_template():
    from langbite.io_managers import json_io_manager
    schema = ConfigFormSchema.model_json_schema()
    choices = schema["$defs"][schema["properties"]["history"]["$ref"].split("/")[-1]]["enum"]
    assert choices[0] == "none" and set(choices[1:]) == set(json_io_manager.load_histories())


def test_the_engine_config_tests_the_target_and_carries_the_judge_and_history():
    plugin = LangBiteEvaluationPlugin()
    cfg = plugin.form_schema_to_internal(ConfigFormSchema(**form(
        useLLMEval=True, judge_model="OpenAIGPT4oMini", judge_api_key="judge-key", history="mcas_faq_api")))
    assert cfg["aiModels"] == ["AISCTarget"]
    assert cfg["judge"] == {"model": "OpenAIGPT4oMini", "api_key": "judge-key"}
    assert cfg["history"] == "mcas_faq_api"
    assert "judge_api_key" not in cfg and "judge_model" not in cfg
    assert plugin.form_schema_to_internal(ConfigFormSchema(**form()))["history"] is None


@pytest.fixture
def platform(monkeypatch):
    import aisc_plugin_interface.system_under_test as sut
    monkeypatch.setattr(sut, "_issue_run_key", lambda pid, path, name: ISSUED)
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "API_KEY_OPENAI"):
        monkeypatch.delenv(k, raising=False)


def test_a_run_reaches_the_target_through_the_platform_and_judges_on_its_own_key(platform, fake_openai):
    """End to end through the plugin: the run key's endpoint is the target, the judge has its own key."""
    from fakes import PROMPTS

    def reply(kwargs):
        return "True" if kwargs["messages"][-1]["content"].startswith("You are evaluating") else "No"
    fake_openai.reply = reply

    plugin = LangBiteEvaluationPlugin()
    plugin._set_artifact_callback(lambda name, content: None)
    plugin.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/system"}).encode())
    plugin.set_input_content("dataset", "\n".join(
        ["\t".join(PROMPTS[0])] + ["\t".join(str(p[k] if p[k] is not None else "") for k in PROMPTS[0]) for p in PROMPTS]
    ).encode())
    result = plugin.evaluate(form(useLLMEval=True, judge_model="OpenAIGPT4oMini", judge_api_key="judge-key"))
    assert result.get("status") != "error", result
    pairs = list(zip(fake_openai.calls, fake_openai.clients))
    tested = [k for c, k in pairs if not c["messages"][-1]["content"].startswith("You are evaluating")]
    judged = [k for c, k in pairs if c["messages"][-1]["content"].startswith("You are evaluating")]
    assert tested and all(k == {"api_key": "aisc-run-key", "base_url": BASE} for k in tested)
    assert judged and all(k["api_key"] == "judge-key" for k in judged)
