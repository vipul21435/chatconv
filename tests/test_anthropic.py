import json
from pathlib import Path

import pytest

from chatconv.formats import FormatError, get_format
from chatconv.losses import summarize
from chatconv.model import Conversation, Image, Text, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"
ANTHROPIC = get_format("anthropic")


def records():
    with open(FIXTURES / "anthropic.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("record", records())
def test_fixture_round_trips_without_loss(record):
    conv = ANTHROPIC.read(record)
    out, losses = ANTHROPIC.write(conv)
    assert losses == []
    assert ANTHROPIC.read(out) == conv


def test_tool_use_and_result_are_linked():
    conv = ANTHROPIC.read(records()[0])
    assert conv.system == [Text("You plan train trips.")]
    assert conv.turns[1].parts[1] == ToolCall(
        "find_trains", {"from": "Ghent", "to": "Lille", "after": "18:00"}, "toolu_x1"
    )
    assert conv.turns[2].parts == [ToolResult([Text("19:12 and 20:47")], "toolu_x1", "find_trains")]
    assert conv.tools[0].parameters["type"] == "object"


def test_base64_image_becomes_data_uri_and_back():
    record = records()[1]
    conv = ANTHROPIC.read(record)
    assert conv.turns[0].parts[0] == Image("data:image/png;base64,iVBORw0KGgo=")
    assert conv.system[0].meta == {"cache_control": {"type": "ephemeral"}}
    assert ANTHROPIC.write(conv)[0] == record


def test_mixed_user_message_splits_into_tool_and_user_turns():
    conv = ANTHROPIC.read(records()[2])
    assert [t.role for t in conv.turns] == ["user", "assistant", "tool", "user"]
    assert conv.turns[2].parts[0].is_error is True


def test_openai_record_to_anthropic():
    openai = get_format("openai")
    conv = openai.read(
        {
            "messages": [
                {
                    "role": "user",
                    "name": "kim",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.org/x.png", "detail": "high"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "ocr", "arguments": "not json"},
                },
                {"role": "function", "name": "ocr", "content": "HELLO"},
            ]
        }
    )
    out, losses = ANTHROPIC.write(conv)
    assert summarize(losses) == {
        "names_dropped": 1,
        "image_detail_dropped": 1,
        "arguments_wrapped": 1,
        "tool_ids_invented": 2,
    }
    use = out["messages"][1]["content"][0]
    assert use == {"type": "tool_use", "id": "call_0", "name": "ocr", "input": {"_raw": "not json"}}
    assert out["messages"][2]["content"][0]["tool_use_id"] == "call_0"
    assert ANTHROPIC.read(out) == ANTHROPIC.fit(conv, losses)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"messages": [{"role": "user", "content": [{"type": "thinking"}]}]}, "unsupported block"),
        ({"messages": [{"role": "bot", "content": "x"}]}, "unknown role"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "file", "file_id": "f1"}}],
                    }
                ]
            },
            "image source",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "name": "f", "input": {}}],
                    }
                ]
            },
            "without id",
        ),
    ],
)
def test_bad_records_raise(record, message):
    with pytest.raises(FormatError, match=message):
        ANTHROPIC.read(record)


def test_system_turn_after_start_is_kept():
    conv = Conversation(turns=[Turn("user", [Text("a")]), Turn("system", [Text("be formal")])])
    out, losses = ANTHROPIC.write(conv)
    assert losses == [] and out["messages"][1]["role"] == "system"
