"""Writes the no-history golden output. Run ONCE on the unchanged engine (8e8afde):
    .venv/bin/python tests/engine/make_golden.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import langbite.llm_services.llm_openai_factory as openai_mod  # noqa: E402
from fakes import RecordingServiceNoHistoryArg, register_fake, run_api  # noqa: E402


class _NoOpenAI:  # the judge is built at TestExecution init; never called with useLLMEval off
    def __init__(self, **_):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))


openai_mod.OpenAI = _NoOpenAI
register_fake("FakeYes", RecordingServiceNoHistoryArg("FakeYes", "Yes"))
register_fake("FakeNo", RecordingServiceNoHistoryArg("FakeNo", "No"))
report = run_api(["FakeYes", "FakeNo"])
for name, df in report.items():
    df.to_csv(HERE / "golden" / f"{name}.csv", index=False)
    print(name, len(df))
