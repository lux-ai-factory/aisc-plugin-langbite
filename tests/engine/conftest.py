import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langbite.llm_services import llm_factory  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_factory():
    """Builders cache one service object each; start every test with none cached
    and drop any fake builders a test registered."""
    before = dict(llm_factory.factory._builders)

    def reset():
        for builder in llm_factory.factory._builders.values():
            for name in list(vars(builder)):
                if name.endswith("instance"):
                    setattr(builder, name, None)

    reset()
    yield
    llm_factory.factory._builders.clear()
    llm_factory.factory._builders.update(before)
    reset()


class FakeOpenAI:
    """Stands in for openai.OpenAI; records every chat.completions.create call."""
    calls = []
    clients = []          # the client (api_key, base_url) each call was made with, in call order
    reply = "Yes"

    def __init__(self, api_key=None, base_url=None, **_ignored):
        self.client = {"api_key": api_key, "base_url": base_url}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeOpenAI.calls.append(kwargs)
        FakeOpenAI.clients.append(self.client)
        content = FakeOpenAI.reply(kwargs) if callable(FakeOpenAI.reply) else FakeOpenAI.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeOllamaClient:
    calls = []
    reply = "Yes"

    def __init__(self, host=None, **_ignored):
        self.host = host

    def chat(self, **kwargs):
        FakeOllamaClient.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=FakeOllamaClient.reply))


@pytest.fixture
def fake_openai(monkeypatch):
    import langbite.llm_services.llm_openai_factory as mod
    FakeOpenAI.calls = []
    FakeOpenAI.clients = []
    FakeOpenAI.reply = "Yes"
    monkeypatch.setattr(mod, "OpenAI", FakeOpenAI)
    return FakeOpenAI


@pytest.fixture
def fake_ollama(monkeypatch):
    import langbite.llm_services.llm_ollama_factory as mod
    FakeOllamaClient.calls = []
    FakeOllamaClient.reply = "Yes"
    monkeypatch.setattr(mod, "Client", FakeOllamaClient)
    return FakeOllamaClient
