import json
from pathlib import Path

import pytest

from chatconv.formats import FormatError, get_format
from chatconv.losses import summarize
from chatconv.model import Conversation, Image, Text, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"
OPENAI = get_format("openai")


def records(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("record", records("openai.jsonl"))
def test_fixture_round_trips_without_loss(record):
    conv = OPENAI.read(record)
    out, losses = OPENAI.write(conv)
    assert losses == []
    assert OPENAI.read(out) == conv


def test_reads_system_tools_and_results():
    conv = OPENAI.read(records("openai.jsonl")[2])
    assert [t.role for t in conv.turns] == ["user", "assistant", "tool", "assistant"]
    calls = conv.turns[1].parts
    assert calls[0] == ToolCall("convert_units", {"value": 12, "from": "mi", "to": "km"}, "call_a1")
    results = conv.turns[2].parts
    assert [r.name for r in results] == ["convert_units", "local_time"]
    assert results[1].content == [Text("14:05")]
    assert [s.name for s in conv.tools] == ["convert_units", "local_time"]


def test_extra_fields_are_kept():
    record = records("openai.jsonl")[0]
    conv = OPENAI.read(record)
    assert conv.meta == {"split": "train"}
    assert conv.turns[-1].meta == {"weight": 1}
    assert OPENAI.write(conv)[0] == record


def test_legacy_function_call_stays_legacy():
    record = records("openai.jsonl")[3]
    conv = OPENAI.read(record)
    assert conv.turns[1].parts[1] == ToolCall("roll_die", {"sides": 6})
    assert conv.turns[2].parts[0] == ToolResult([Text("4")], None, "roll_die")
    out, losses = OPENAI.write(conv)
    assert losses == []
    assert out["messages"][1]["function_call"]["name"] == "roll_die"
    assert out["messages"][2]["role"] == "function"


def test_missing_ids_are_invented_and_linked():
    conv = Conversation(
        turns=[
            Turn("user", [Text("two lookups")]),
            Turn("assistant", [ToolCall("a", {}), ToolCall("b", {"x": 1})]),
            Turn("tool", [ToolResult([Text("B")], name="b"), ToolResult([Text("A")])]),
        ]
    )
    out, losses = OPENAI.write(conv)
    assert summarize(losses) == {"tool_ids_invented": 2}
    msgs = out["messages"]
    assert [c["id"] for c in msgs[1]["tool_calls"]] == ["call_0", "call_1"]
    assert [m["tool_call_id"] for m in msgs[2:]] == ["call_1", "call_0"]
    back = OPENAI.read(out)
    assert back == OPENAI.fit(conv, losses)
    assert [r.name for r in back.turns[2].parts] == ["b", "a"]


def test_images_in_tool_results_and_error_flags_are_lost():
    conv = Conversation(
        turns=[
            Turn("user", [Text("screenshot please")]),
            Turn("assistant", [ToolCall("shot", {}, "c1")]),
            Turn(
                "tool",
                [ToolResult([Image("data:image/png;base64,AAAA")], "c1", "shot", is_error=True)],
            ),
        ]
    )
    out, losses = OPENAI.write(conv)
    assert summarize(losses) == {"images_dropped": 1, "tool_error_flag_dropped": 1}
    assert OPENAI.read(out) == OPENAI.fit(conv, losses)


def test_text_after_calls_is_reordered():
    conv = Conversation(
        turns=[Turn("user", [Text("q")]), Turn("assistant", [ToolCall("f", {}, "c1"), Text("ok")])]
    )
    out, losses = OPENAI.write(conv)
    assert summarize(losses) == {"parts_reordered": 1}
    assert OPENAI.read(out) == OPENAI.fit(conv, losses)


@pytest.mark.parametrize(
    "record, message",
    [
        ({}, "no messages"),
        ({"messages": [{"role": "robot", "content": "x"}]}, "unknown role"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "tool_call_id"),
        (
            {"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]},
            "unsupported content part",
        ),
    ],
)
def test_bad_records_raise(record, message):
    with pytest.raises(FormatError, match=message):
        OPENAI.read(record)
