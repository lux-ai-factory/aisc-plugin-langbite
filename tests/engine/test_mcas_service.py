"""MCASChat: LangBiTe testing the real MCAS-lite /chat over HTTP, with or without a history."""
from types import SimpleNamespace

import pytest

from fakes import run_api
from langbite.io_managers import json_io_manager, secrets
from langbite.llm_services import llm_factory

URL = "http://mcas.test:8500"
HISTORY = [{"role": "user", "content": "Can I ask a person to review a rejection?"},
           {"role": "assistant", "content": "Yes, at no cost (POL-FAIR-002)."}]


class FakePost:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.calls = status, body, []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        body = self.body if self.body is not None else {"answer": "No, it is not an input (POL-FAIR-001).",
                                                        "policy_refs": ["POL-FAIR-001"]}
        return SimpleNamespace(status_code=self.status, json=lambda: body, text=str(body))


@pytest.fixture
def post(monkeypatch):
    import langbite.llm_services.llm_mcas_service as mod
    fake = FakePost()
    monkeypatch.setattr(mod.requests, "post", fake)
    return fake


def mcas():
    return llm_factory.factory.create("MCASChat", mcas_url=URL)


def test_the_url_comes_from_the_environment_with_a_local_default(monkeypatch):
    monkeypatch.delenv("MCAS_URL", raising=False)
    assert secrets.load_api_keys()["mcas_url"] == "http://localhost:8500"
    monkeypatch.setenv("MCAS_URL", URL)
    assert secrets.load_api_keys()["mcas_url"] == URL


def test_without_history_it_posts_the_question_alone(post):
    assert mcas().execute_prompt("Is my nationality an input?") == "No, it is not an input (POL-FAIR-001)."
    assert post.calls[0]["url"] == f"{URL}/chat"
    assert post.calls[0]["json"] == {"question": "Is my nationality an input?", "history": []}


def test_with_history_it_posts_the_turns(post):
    mcas().execute_prompt("Is my nationality an input?", history=HISTORY)
    assert post.calls[0]["json"]["history"] == HISTORY


def test_a_system_turn_is_rejected_before_any_request(post):
    with pytest.raises(NotImplementedError, match="_api"):
        mcas().execute_prompt("Q?", history=[{"role": "system", "content": "S"}] + HISTORY)
    assert post.calls == []


def test_a_grounding_refusal_is_returned_as_the_answer(post):
    post.status = 502
    post.body = {"detail": {"error": "answer_not_grounded", "reason": "no policy clause covers it"}}
    answer = mcas().execute_prompt("Should I buy shares?")
    assert answer.startswith("Refused by MCAS")
    assert "no policy clause covers it" in answer


@pytest.mark.parametrize("status", [422, 503, 500])
def test_other_errors_raise(post, status):
    post.status = status
    post.body = {"detail": "x"}
    with pytest.raises(RuntimeError, match=str(status)):
        mcas().execute_prompt("Q?")


@pytest.mark.parametrize("name", ["mcas_faq", "mcas_rejected_applicant", "mcas_pressure"])
def test_api_variants_are_the_same_turns_without_the_system_prompt(name):
    histories = json_io_manager.load_histories()
    assert histories[f"{name}_api"] == histories[name][1:]
    mcas().validate_history(histories[f"{name}_api"])            # accepted
    with pytest.raises(NotImplementedError):
        mcas().validate_history(histories[name])                  # system version refused


def test_end_to_end_run_sends_the_history_to_mcas(post, fake_openai):
    post.body = {"answer": "Yes, you can (POL-FAIR-002).", "policy_refs": ["POL-FAIR-002"]}
    report = run_api(["MCASChat"], history="mcas_rejected_applicant_api")
    expected = json_io_manager.load_histories()["mcas_rejected_applicant_api"]
    assert len(post.calls) == 4
    assert all(call["json"]["history"] == expected for call in post.calls)
    assert set(report["global_eval"]["Model"]) == {"MCASChat+mcas_rejected_applicant_api"}
