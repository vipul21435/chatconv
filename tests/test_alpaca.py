import json
from pathlib import Path

import pytest

from chatconv.formats import get_format
from chatconv.losses import Policy, summarize
from chatconv.model import Conversation, Text, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"


def records(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("fmt, fixture", [("alpaca", "alpaca.jsonl")])
def test_fixtures_round_trip(fmt, fixture):
    f = get_format(fmt)
    for record in records(fixture):
        conv = f.read(record)
        out, losses = f.write(conv)
        assert losses == []
        assert f.read(out) == conv


def test_alpaca_reads_input_history_and_system():
    alpaca = get_format("alpaca")
    first, _, multi = records("alpaca.jsonl")
    conv = alpaca.read(first)
    assert conv.turns[0].parts == [Text("Give the plural of the word."), Text("cactus")]
    conv = alpaca.read(multi)
    assert [t.role for t in conv.turns] == ["user", "assistant", "user", "assistant"]
    assert conv.system == [Text("Answer in one word.")] and conv.meta == {"source": "handmade"}


def tool_conv():
    return Conversation(
        system=[Text("Be exact.")],
        turns=[
            Turn("user", [Text("Square root of 2?")]),
            Turn("assistant", [ToolCall("sqrt", {"x": 2}, "c1")]),
            Turn("tool", [ToolResult([Text("1.4142")], "c1", "sqrt")]),
            Turn("assistant", [Text("About 1.4142.")]),
        ],
    )


def test_alpaca_tools_as_text_and_alternation():
    out, losses = get_format("alpaca").write(tool_conv())
    assert summarize(losses) == {"tool_calls_as_text": 1, "tool_results_as_text": 1}
    assert out["history"] == [["Square root of 2?", '{"name": "sqrt", "arguments": {"x": 2}}']]
    assert out["instruction"] == "1.4142" and out["output"] == "About 1.4142."
    out, losses = get_format("alpaca").write(tool_conv(), Policy(tools="drop"))
    assert summarize(losses) == {
        "tool_calls_dropped": 1,
        "tool_results_dropped": 1,
        "turns_merged": 1,
    }
    assert "history" not in out
    assert out["output"] == "About 1.4142."


def test_alpaca_adds_turns_for_order():
    conv = Conversation(turns=[Turn("assistant", [Text("hello")]), Turn("user", [Text("hi")])])
    out, losses = get_format("alpaca").write(conv)
    assert summarize(losses) == {"turns_added": 2}
    assert out == {"instruction": "hi", "input": "", "output": "", "history": [["", "hello"]]}


