"""T1, T2: history templates file and its validation (C1)."""
import pytest

from langbite.io_managers import json_io_manager
from langbite.model.conversation_history import parse_histories

GOOD = {
    "name": "small_talk",
    "description": "d",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ],
}


def _with(**changes):
    item = {**GOOD, **changes}
    return [item]


def test_t1_packaged_histories_load_by_name():
    histories = json_io_manager.load_histories()
    assert len(histories) >= 2
    for name, messages in histories.items():
        assert isinstance(name, str) and name
        assert messages and all({"role", "content"} <= set(m) for m in messages)


def test_t1_parse_returns_messages_keyed_by_name():
    assert parse_histories([GOOD]) == {"small_talk": GOOD["messages"]}


def test_t1_only_role_and_content_are_kept():
    item = _with(messages=[{"role": "system", "content": "x", "note": "extra"}])
    assert parse_histories(item) == {"small_talk": [{"role": "system", "content": "x"}]}


@pytest.mark.parametrize("bad", [
    _with(messages=[{"role": "tool", "content": "x"}]),                              # bad role
    _with(messages=[{"role": "system", "content": "  "}]),                            # empty content
    _with(messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "u"}]),  # ends with user
    _with(messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"},
                    {"role": "system", "content": "late"}]),                          # system not first
    _with(messages=[]),                                                               # no messages
    _with(name=""),                                                                   # no name
    [GOOD, GOOD],                                                                     # duplicate name
])
def test_t2_invalid_templates_raise(bad):
    with pytest.raises(ValueError):
        parse_histories(bad)
