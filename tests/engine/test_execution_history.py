"""T11-T13: history through a full run, labels, fail-fast, judge isolation (C8)."""
import pytest

from fakes import RecordingService, register_fake, run_api
from langbite.io_managers import json_io_manager


@pytest.fixture
def history():
    name, messages = next(iter(json_io_manager.load_histories().items()))
    return name, messages


def test_t11_history_reaches_every_prompt_and_labels_rows(fake_openai, history):
    name, messages = history
    service = register_fake("FakeYes", RecordingService("FakeYes"))
    report = run_api(["FakeYes"], history=name)

    assert len(service.calls) == 4                          # 2 prompts x 2 communities
    assert all(call["history"] == messages for call in service.calls)
    label = f"FakeYes+{name}"
    assert set(report["responses"]["Model"]) == {label}
    assert set(report["evaluations"]["Model"]) == {label}
    assert set(report["global_eval"]["Model"]) == {label}


def test_t11_no_history_keeps_plain_label(fake_openai):
    service = register_fake("FakeYes", RecordingService("FakeYes"))
    report = run_api(["FakeYes"])
    assert all(call["history"] is None for call in service.calls)
    assert set(report["responses"]["Model"]) == {"FakeYes"}


def test_t12_unsupported_backend_fails_before_first_prompt(fake_openai, history):
    name, _ = history
    service = register_fake("FakeOld", RecordingService("FakeOld", accepts_history=False))
    with pytest.raises(NotImplementedError):
        run_api(["FakeOld"], history=name)
    assert service.calls == []


def test_t13_judge_never_sees_history_even_on_its_own_cached_object(fake_openai, history):
    """Model under test is OpenAIGPT4, the judge's own key, so both share one cached
    service object. Answers are "No" so every check fails and goes to the judge."""
    name, messages = history

    def reply(kwargs):
        content = kwargs["messages"][-1]["content"]
        return "True" if content.startswith("You are evaluating") else "No"

    fake_openai.reply = reply
    run_api(["OpenAIGPT4"], history=name, use_llm_eval=True)

    judge = [c for c in fake_openai.calls if c["messages"][-1]["content"].startswith("You are evaluating")]
    tested = [c for c in fake_openai.calls if c not in judge]
    assert judge, "the judge was never called, so the test proves nothing"
    assert all(len(c["messages"]) == 1 for c in judge)
    assert tested and all(c["messages"][:-1] == messages for c in tested)
