import json
from pathlib import Path

import pytest

from chatconv.formats import FormatError, get_format
from chatconv.losses import Policy, summarize
from chatconv.model import Conversation, Image, Text, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"
GEMINI = get_format("gemini")


def records():
    with open(FIXTURES / "gemini.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("record", records())
def test_fixture_round_trips_without_loss(record):
    conv = GEMINI.read(record)
    out, losses = GEMINI.write(conv)
    assert losses == []
    assert GEMINI.read(out) == conv


def test_reads_calls_responses_and_images():
    conv = GEMINI.read(records()[1])
    assert conv.turns[1].parts == [
        ToolCall("price", {"item": "pencil"}),
        ToolCall("price", {"item": "eraser"}),
    ]
    assert conv.turns[2].parts[1] == ToolResult([Text("0.25")], None, "price")
    conv = GEMINI.read(records()[2])
    assert conv.turns[0].parts[0] == Image("data:image/jpeg;base64,/9j/4AAQ")
    assert conv.turns[0].parts[1] == Image(
        "https://example.org/chart.png", None, {"mimeType": "image/png"}
    )
    assert conv.turns[2].parts[0].content == [Text('{"left": 3, "right": 5}')]


def test_sharegpt_results_get_names_from_their_calls():
    sharegpt = get_format("sharegpt")
    conv = sharegpt.read(
        {
            "conversations": [
                {"from": "human", "value": "x"},
                {"from": "function_call", "value": '{"name": "lookup", "arguments": {}}'},
                {"from": "observation", "value": "found"},
            ]
        }
    )
    out, losses = GEMINI.write(conv)
    assert summarize(losses) == {"tool_result_names_inferred": 1}
    assert out["contents"][2]["parts"][0]["functionResponse"]["name"] == "lookup"


def test_system_turns_policy():
    conv = Conversation(turns=[Turn("user", [Text("a")]), Turn("system", [Text("b")])])
    out, losses = GEMINI.write(conv)
    assert summarize(losses) == {"system_turns_as_user": 1}
    out, losses = GEMINI.write(conv, Policy(system_turns="drop"))
    assert summarize(losses) == {"system_turns_dropped": 1}
    assert len(out["contents"]) == 1


@pytest.mark.parametrize(
    "record, message",
    [
        (
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "AA"}}],
                    }
                ]
            },
            "unsupported inlineData",
        ),
        ({"contents": [{"role": "robot", "parts": []}]}, "unknown role"),
        (
            {"contents": [{"role": "user", "parts": [{"text": "a", "functionCall": {}}]}]},
            "exactly one",
        ),
        ({"contents": [], "tools": [{"googleSearch": {}}]}, "unsupported tool"),
    ],
)
def test_bad_records_raise(record, message):
    with pytest.raises(FormatError, match=message):
        GEMINI.read(record)
