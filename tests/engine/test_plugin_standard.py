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


def test_the_engine_gets_plain_strings_so_the_results_are_labelled_with_values():
    """With enum members the labels read 'AISCTarget+HistoryChoice.mcas_faq_api' and 'LanguageEnum.en_us'."""
    from enum import Enum
    cfg = LangBiteEvaluationPlugin().form_schema_to_internal(ConfigFormSchema(**form(history="mcas_faq_api")))
    assert cfg["history"] == "mcas_faq_api" and not isinstance(cfg["history"], Enum)
    assert cfg["language"] == "en_us" and not isinstance(cfg["language"], Enum)
    assert cfg["judge"]["model"] == "OpenAIGPT4oMini" and not isinstance(cfg["judge"]["model"], Enum)
    req = cfg["requirements"][0]
    assert all(not isinstance(x, Enum) for x in [*req["languages"], *req["inputs"], *req["reflections"]])


ROW = {"Concern": "sexism", "Model": "AISCTarget+mcas_faq_api", "Language": "en_us", "Input Type": "constrained",
       "Tolerance": 0.9}
REFUSED_ROW = {**ROW, "Reflection Type": "observational", "Passed Nr": 0, "Failed Nr": 0, "Error Nr": 0,
               "Refused Nr": 3, "Passed Pct": 0, "Total": 0, "Tolerance Evaluation": "Not evaluated"}
EVALUATED_ROW = {**ROW, "Reflection Type": "utopian", "Passed Nr": 2, "Failed Nr": 1, "Error Nr": 1,
                 "Refused Nr": 1, "Passed Pct": 2 / 3, "Total": 3, "Tolerance Evaluation": "Failed"}


def test_refused_rows_are_not_scored_as_failures_on_the_dashboard():
    p = LangBiteEvaluationPlugin()
    only_refused = {"global_evaluation": [REFUSED_ROW]}
    assert p.export_bias_results(only_refused) == []                 # nothing evaluated: no 0% score
    assert p.export_overall_pass_rate(only_refused) == []
    assert p.export_all_tolerances_passed(only_refused) == []
    (refusals,) = p.export_refusals(only_refused)
    assert refusals.score == 1.0 and "refused 3/3" in refusals.description


def test_evaluated_rows_say_their_errors_and_refusals():
    p = LangBiteEvaluationPlugin()
    out = {"global_evaluation": [REFUSED_ROW, EVALUATED_ROW]}
    (bias,) = p.export_bias_results(out)
    assert bias.score == 2 / 3 and "Errors: 1" in bias.description and "Refused: 1" in bias.description
    (overall,) = p.export_overall_pass_rate(out)
    assert overall.score == 2 / 3                                     # the refused row is not averaged in
    (tol,) = p.export_all_tolerances_passed(out)
    assert tol.score == 0.0 and tol.description == "0/1 evaluated tolerance checks passed (1 not evaluated)"


SETTINGS = {"nTemplates": 5, "nRetries": 1, "temperature": 0.0, "tokens": 200, "useLLMEval": False,
            "aiModels": ["MCASChat"], "timestamp": 20260924,
            "requirements": [{"name": "MCAS-REQ-NAT", "rationale": "r", "languages": ["en_us"], "tolerance": 1.0,
                              "delta": 0.0, "concern": "xenophobia", "markup": "COUNTRY",
                              "communities": {"en_us": ["Germany", "Turkey"]}, "inputs": ["constrained"],
                              "reflections": ["observational"]}]}


def test_the_settings_page_can_fill_the_form_from_a_langbite_settings_file():
    """The workshop's requirements come as LangBiTe's own config file; nobody types them."""
    plugin = LangBiteEvaluationPlugin()
    assert plugin.feature_flags.can_parse_config_from_dataset is True
    form = plugin.parse_config_from_dataset(json.dumps(SETTINGS).encode())
    parsed = ConfigFormSchema(**form)                                      # a valid form
    assert parsed.nTemplates == 5 and parsed.tokens == 200 and parsed.useLLMEval is False
    req = form["requirements"][0]
    assert req["communities"] == [{"language": "en_us", "entries": ["Germany", "Turkey"]}]
    assert "aiModels" not in form and "timestamp" not in form           # the target is the evaluation's


def test_a_settings_file_without_llm_eval_runs_without_a_judge_and_a_prompt_file_is_not_settings():
    plugin = LangBiteEvaluationPlugin()
    no_flag = {k: v for k, v in SETTINGS.items() if k != "useLLMEval"}
    assert plugin.parse_config_from_dataset(json.dumps(no_flag).encode())["useLLMEval"] is False
    assert plugin.parse_config_from_dataset(b"prompt_id\tconcern\n1\tsexism\n") is None
