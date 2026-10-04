"""The AISC standard (aisc docs/superpowers/target-standard-2026-10-04/01-plan.md): the model under test
is the evaluation's target, reached only through AISC_TARGET_* (set by the platform for the run); the
judge is a tool model with its own model and key, on its own client, never the target's."""
import pytest

from fakes import run_api
from langbite.llm_services import llm_factory
from langbite.io_managers import json_io_manager
from langbite.oracles.sentiment_analyzer_oracle import parse_verdict

TARGET = {"AISC_TARGET_BASE_URL": "http://platform:8000/internal/facade/p/mcas-chat/openai/v1",
          "AISC_TARGET_API_KEY": "aisc-run-key", "AISC_TARGET_MODEL": "mcas-chat"}
JUDGE = {"model": "OpenAIGPT4oMini", "api_key": "judge-key"}


@pytest.fixture
def target_env(monkeypatch):
    for k, v in TARGET.items():
        monkeypatch.setenv(k, v)
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "API_KEY_OPENAI"):
        monkeypatch.delenv(k, raising=False)


def is_judge(call):
    return call["messages"][-1]["content"].startswith("You are evaluating")


def test_the_target_is_reached_through_aisc_target_only(fake_openai, target_env):
    fake_openai.reply = "Yes"
    run_api(["AISCTarget"])
    assert fake_openai.calls and all(c == {"api_key": "aisc-run-key", "base_url": TARGET["AISC_TARGET_BASE_URL"]}
                                     for c in fake_openai.clients)
    assert {c["model"] for c in fake_openai.calls} == {"mcas-chat"}


def test_the_target_without_its_variables_is_refused_before_the_first_prompt(fake_openai, monkeypatch):
    for k in TARGET:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="AISC_TARGET_BASE_URL"):
        run_api(["AISCTarget"])
    assert fake_openai.calls == []


def test_the_judge_has_its_own_model_and_key_and_never_the_targets(fake_openai, target_env):
    def reply(kwargs):
        return "True" if is_judge(kwargs) else "No"          # every check fails, so each goes to the judge
    fake_openai.reply = reply
    run_api(["AISCTarget"], use_llm_eval=True, judge=JUDGE)
    pairs = list(zip(fake_openai.calls, fake_openai.clients))
    judge = [(c, k) for c, k in pairs if is_judge(c)]
    tested = [(c, k) for c, k in pairs if not is_judge(c)]
    assert judge and tested
    assert all(k == {"api_key": "judge-key", "base_url": "https://api.openai.com/v1"} and c["model"] == "gpt-4o-mini"
               for c, k in judge)
    assert all(k["api_key"] == "aisc-run-key" for c, k in tested)


def test_llm_eval_without_a_judge_key_is_refused_before_the_first_prompt(fake_openai, target_env):
    with pytest.raises(ValueError, match="judge"):
        run_api(["AISCTarget"], use_llm_eval=True, judge={"model": "OpenAIGPT4oMini", "api_key": ""})
    assert fake_openai.calls == []


def test_no_judge_is_made_without_llm_eval(fake_openai, target_env):
    fake_openai.reply = "No"
    run_api(["AISCTarget"])
    assert not [c for c in fake_openai.calls if is_judge(c)]


@pytest.mark.parametrize("text, verdict", [("True", True), ("false", False), (" True.\n", True), ("FALSE", False)])
def test_a_verdict_is_read_strictly(text, verdict):
    assert parse_verdict(text) is verdict


@pytest.mark.parametrize("text", ["__import__('os').system('id')", "Maybe", "True or False", "", "1"])
def test_anything_else_is_refused_never_evaluated(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_the_mcas_only_backend_is_gone():
    assert "MCASChat" not in llm_factory.factory._builders
    assert "MCAS" not in {f["provider"].upper() for f in json_io_manager.load_factories()}
    from langbite.io_managers import secrets
    assert "mcas_url" not in secrets.load_api_keys()


def test_the_api_histories_say_what_they_are_for_without_naming_a_removed_backend():
    raw = json_io_manager.load_histories_raw() if hasattr(json_io_manager, "load_histories_raw") else None
    import json, pathlib
    items = json.loads((pathlib.Path(json_io_manager.__file__).parents[1] / "resources" / "histories.json").read_text())
    for item in items:
        assert "MCASChat" not in item["description"], item["name"]
