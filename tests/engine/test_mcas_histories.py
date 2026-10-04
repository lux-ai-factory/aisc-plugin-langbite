"""The three MCAS-lite assistant histories: they load, carry the full MCAS policy corpus in
the system prompt, and every assistant turn cites only clauses that exist."""
import re

import pytest

from langbite.io_managers import json_io_manager

MCAS_NAMES = ["mcas_faq", "mcas_rejected_applicant", "mcas_pressure"]
CLAUSE_IDS = {"POL-ELIG-001", "POL-ELIG-002", "POL-ELIG-003", "POL-ELIG-004", "POL-DEC-001",
              "POL-DEC-002", "POL-FAIR-001", "POL-FAIR-002", "POL-DATA-001", "POL-DATA-002"}
CITATION = re.compile(r"\bPOL-[A-Z]+-\d{3}\b")


@pytest.fixture(scope="module")
def histories():
    return json_io_manager.load_histories()


@pytest.mark.parametrize("name", MCAS_NAMES)
def test_mcas_history_loads_with_full_corpus(histories, name):
    messages = histories[name]
    assert messages[0]["role"] == "system"
    assert set(CITATION.findall(messages[0]["content"])) == CLAUSE_IDS
    assert len(messages) >= 5                                # system + at least two exchanges


@pytest.mark.parametrize("name", MCAS_NAMES)
def test_mcas_assistant_turns_cite_real_clauses(histories, name):
    for message in histories[name]:
        if message["role"] == "assistant":
            cited = set(CITATION.findall(message["content"]))
            assert cited, message["content"]
            assert cited <= CLAUSE_IDS


def test_mcas_histories_share_one_system_prompt(histories):
    assert len({histories[n][0]["content"] for n in MCAS_NAMES}) == 1


@pytest.mark.parametrize("name", MCAS_NAMES)
def test_api_variants_are_the_same_turns_without_the_system_prompt(histories, name):
    """For a target that supplies its own system prompt (MCAS /chat refuses a system turn)."""
    assert histories[f"{name}_api"] == histories[name][1:]
    assert all(m["role"] != "system" for m in histories[f"{name}_api"])
