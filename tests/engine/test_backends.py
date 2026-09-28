"""T6-T9: what each backend sends, with and without a history (C2-C6)."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from langbite.llm_services import llm_factory

HISTORY = [
    {"role": "system", "content": "You are a support agent."},
    {"role": "user", "content": "My order is late."},
    {"role": "assistant", "content": "Sorry to hear that."},
]
SUFFIX = " Do not use carry returns in your response."


# ---------- T6, T7: OpenAI ----------

def test_t6_openai_without_history_sends_exactly_today(fake_openai):
    service = llm_factory.factory.create("OpenAIGPT4o", openai_api_key="k")
    assert service.execute_prompt("Is the sky blue?") == "Yes"
    assert fake_openai.calls == [{
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Is the sky blue?" + SUFFIX}],
    }]


def test_t7_openai_with_history_prepends_it(fake_openai):
    service = llm_factory.factory.create("OpenAIGPT4o", openai_api_key="k")
    original = [dict(m) for m in HISTORY]
    service.execute_prompt("Q1", history=HISTORY)
    service.execute_prompt("Q2", history=HISTORY)
    first, second = (c["messages"] for c in fake_openai.calls)
    assert first == HISTORY + [{"role": "user", "content": "Q1" + SUFFIX}]
    assert second == HISTORY + [{"role": "user", "content": "Q2" + SUFFIX}]
    assert HISTORY == original                      # template not modified
    assert first[:-1] is not HISTORY                # a copy is sent, not the template


# ---------- T8: Ollama ----------

def _ollama():
    return llm_factory.factory.create("OLlamaLlama3.2", ollama_url="http://x")


def test_t8_ollama_without_history_sends_exactly_today(fake_ollama):
    _ollama().execute_prompt("Is the sky blue?")
    assert fake_ollama.calls[0]["messages"] == [{"role": "user", "content": "Is the sky blue?"}]


def test_t8_ollama_with_history_prepends_it(fake_ollama):
    original = [dict(m) for m in HISTORY]
    service = _ollama()
    service.execute_prompt("Q1", history=HISTORY)
    service.execute_prompt("Q2", history=HISTORY)
    assert fake_ollama.calls[0]["messages"] == HISTORY + [{"role": "user", "content": "Q1"}]
    assert fake_ollama.calls[1]["messages"] == HISTORY + [{"role": "user", "content": "Q2"}]
    assert HISTORY == original


def test_t8_ollama_still_strips_think(fake_ollama):
    fake_ollama.reply = "<think>hmm</think> Yes"
    assert _ollama().execute_prompt("Q", history=HISTORY) == "Yes"


# ---------- T9: backends without history support ----------

def _huggingface(monkeypatch):
    from langbite.llm_services.llm_huggingface_factory import HuggingFaceService
    monkeypatch.setattr(HuggingFaceService, "query", lambda self, payload: [{"generated_text": "Yes"}])
    return llm_factory.factory.create("HuggingFaceMixtral8x7B01Instruct", huggingface_api_key="k")


def _replicate(monkeypatch):
    import langbite.llm_services.llm_replicate_service as mod
    monkeypatch.setattr(mod.replicate, "run", lambda model, input: ["Yes"])
    return mod.ReplicateService("k", "some/model")


def _example(monkeypatch, fake_openai):
    import langbite.llm_services.plugins.llm_example_service as mod
    monkeypatch.setattr(mod, "OpenAI", fake_openai)
    return mod.ExampleService()


@pytest.mark.parametrize("make", ["huggingface", "replicate", "example"])
def test_t9_unsupported_backends(make, monkeypatch, fake_openai):
    service = {"huggingface": lambda: _huggingface(monkeypatch),
               "replicate": lambda: _replicate(monkeypatch),
               "example": lambda: _example(monkeypatch, fake_openai)}[make]()
    assert service.execute_prompt("Q") == "Yes"                 # unchanged without history
    assert service.execute_prompt("Q", history=[]) == "Yes"     # empty history = none
    with pytest.raises(NotImplementedError):
        service.execute_prompt("Q", history=HISTORY)
    with pytest.raises(NotImplementedError):
        service.validate_history(HISTORY)


# ---------- T9: GPT4All (spike S1: system prompt only) ----------

class _FakeGPT4All:
    def __init__(self):
        self.sessions = []

    @contextmanager
    def chat_session(self, system_prompt=None):
        self.sessions.append(system_prompt)
        yield self

    def generate(self, prompt):
        return "<|assistant|>Yes\n"


def _gpt4all():
    from langbite.llm_services.llm_gpt4all_service import GPT4AllService
    service = GPT4AllService.__new__(GPT4AllService)   # skip the real model download
    service.provider = "GPT4All"
    service.llm = _FakeGPT4All()
    return service


def test_t9_gpt4all_without_history_unchanged():
    service = _gpt4all()
    assert service.execute_prompt("Q") == "Yes"
    assert service.llm.sessions == [None]


def test_t9_gpt4all_accepts_single_system_message():
    service = _gpt4all()
    service.execute_prompt("Q", history=[{"role": "system", "content": "Be kind."}])
    assert service.llm.sessions == ["Be kind."]


def test_t9_gpt4all_rejects_earlier_turns():
    service = _gpt4all()
    with pytest.raises(NotImplementedError):
        service.execute_prompt("Q", history=HISTORY)
    with pytest.raises(NotImplementedError):
        service.validate_history(HISTORY)
