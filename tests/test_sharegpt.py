import json
from pathlib import Path

import pytest

from chatconv.formats import FormatError, get_format
from chatconv.losses import summarize
from chatconv.model import Conversation, Image, Text, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"
SHAREGPT = get_format("sharegpt")
OPENAI = get_format("openai")


def records(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("record", records("sharegpt.jsonl"))
def test_fixture_round_trips_without_loss(record):
    conv = SHAREGPT.read(record)
    out, losses = SHAREGPT.write(conv)
    # the third fixture has two observations in a row, which the strict order rule flags
    assert {loss.code for loss in losses} <= {"order_broken"}
    assert SHAREGPT.read(out) == conv


def test_reads_system_and_parallel_calls():
    conv = SHAREGPT.read(records("sharegpt.jsonl")[2])
    assert conv.system == [Text("Use the tools when needed.")]
    assert [t.role for t in conv.turns] == ["user", "assistant", "tool", "assistant"]
    assert conv.turns[1].parts == [
        ToolCall("weather", {"city": "Lima"}),
        ToolCall("weather", {"city": "Quito"}),
    ]
    assert len(conv.turns[2].parts) == 2  # two observations, one tool turn


def test_sharegpt_to_openai_invents_linked_ids():
    conv = SHAREGPT.read(records("sharegpt.jsonl")[2])
    out, losses = OPENAI.write(conv)
    assert summarize(losses) == {"tool_ids_invented": 2}
    msgs = out["messages"]
    assert [c["id"] for c in msgs[2]["tool_calls"]] == ["call_0", "call_1"]
    assert [m["tool_call_id"] for m in msgs[3:5]] == ["call_0", "call_1"]


def test_openai_to_sharegpt_reports_each_loss():
    conv = OPENAI.read(
        {
            "messages": [
                {
                    "role": "user",
                    "name": "ravi",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {"type": "image_url", "image_url": {"url": "https://example.org/cat.jpg"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "c7",
                            "type": "function",
                            "function": {"name": "label_image", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "c7", "content": "cat"},
                {"role": "assistant", "content": "A cat."},
            ]
        }
    )
    out, losses = SHAREGPT.write(conv)
    assert summarize(losses) == {
        "images_dropped": 1,
        "names_dropped": 1,
        "turn_split": 1,
        "tool_ids_dropped": 2,
        "tool_result_names_dropped": 1,
        "order_broken": 1,
    }
    assert [e["from"] for e in out["conversations"]] == [
        "human",
        "gpt",
        "function_call",
        "observation",
        "gpt",
    ]
    assert SHAREGPT.read(out) == SHAREGPT.fit(conv, losses)


def test_image_placeholder_policy():
    from chatconv.losses import Policy

    conv = Conversation(turns=[Turn("user", [Image("https://example.org/a.png"), Text("what?")])])
    out, losses = SHAREGPT.write(conv, Policy(images="placeholder"))
    assert summarize(losses) == {"images_as_placeholder": 1, "text_joined": 1}
    assert out["conversations"][0]["value"] == "<image>\nwhat?"


def test_error_flag_and_result_meta():
    conv = Conversation(
        turns=[
            Turn("assistant", [ToolCall("f", "not json")]),
            Turn("tool", [ToolResult([Text("boom")], is_error=True, meta={"x_ms": 3})]),
        ]
    )
    out, losses = SHAREGPT.write(conv)
    assert summarize(losses) == {"tool_error_flag_dropped": 1, "order_broken": 1}
    assert out["conversations"][1] == {"x_ms": 3, "from": "observation", "value": "boom"}
    assert SHAREGPT.read(out) == SHAREGPT.fit(conv, losses)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"messages": []}, "no conversations"),
        ({"conversations": [{"from": "alien", "value": "x"}]}, "unknown speaker"),
        ({"conversations": [{"from": "human", "value": 3}]}, "must be a string"),
        ({"conversations": [{"from": "function_call", "value": "{"}]}, "not JSON"),
        ({"conversations": [], "tools": "nope"}, "not JSON"),
    ],
)
def test_bad_records_raise(record, message):
    with pytest.raises(FormatError, match=message):
        SHAREGPT.read(record)


def test_tagged_tool_calls_and_tool_entries():
    record = {
        "conversations": [
            {"from": "human", "value": "Book a court and a coach."},
            {
                "from": "gpt",
                "value": '<tool_call>\n{"name": "book_court", "arguments": {"hour": 9}}\n'
                '</tool_call>\n<tool_call>\n{"name": "book_coach", "arguments": {}}\n</tool_call>',
            },
            {"from": "tool", "value": "<tool_response>court 3</tool_response>"},
        ],
        "tools": '[{"type": "function", "function": {"name": "book_court"}}]',
    }
    conv = SHAREGPT.read(record)
    assert conv.turns[1].parts == [ToolCall("book_court", {"hour": 9}), ToolCall("book_coach", {})]
    assert conv.turns[2].parts == [ToolResult([Text("<tool_response>court 3</tool_response>")])]
    assert conv.tools[0].name == "book_court"


@pytest.mark.parametrize(
    "value",
    [
        "<tool_call>{not json}</tool_call>",
        'Sure. <tool_call>{"name": "f"}</tool_call>',
        "<tool_call>[1]</tool_call>",
    ],
)
def test_text_that_is_not_only_tagged_calls_stays_text(value):
    conv = SHAREGPT.read({"conversations": [{"from": "gpt", "value": value}]})
    assert conv.turns[0].parts == [Text(value)]


def test_python_literal_tagged_calls_are_parsed():
    body = "{'arguments': {'q': ['a', \"b's\"], 'n': (1, 2), 'on': True}, 'name': 'find'}"
    value = f"<tool_call>\n{body}\n</tool_call>"
    conv = SHAREGPT.read({"conversations": [{"from": "gpt", "value": value}]})
    assert conv.turns[0].parts == [ToolCall("find", {"q": ["a", "b's"], "n": [1, 2], "on": True})]
    assert conv.notes == []


@pytest.mark.parametrize(
    "value",
    [
        "<tool_call>{'name': 'f', 'arguments': {1, 2}}</tool_call>",  # a set is not JSON
        "<tool_call>__import__('os')</tool_call>",
        "<tool_call>{'name': f}</tool_call>",
    ],
)
def test_unparseable_tagged_calls_stay_text_with_a_read_note(value):
    conv = SHAREGPT.read({"conversations": [{"from": "gpt", "value": value}]})
    assert conv.turns[0].parts == [Text(value)]
    assert conv.notes == ["tagged_calls_unparsed"]


def test_read_notes_are_reported(tmp_path, capsys):
    from chatconv.cli import main

    src = tmp_path / "n.jsonl"
    src.write_text(
        json.dumps(
            {"conversations": [{"from": "gpt", "value": "<tool_call>{'name': f}</tool_call>"}]}
        )
        + "\n"
    )
    main(["losses", str(src), "--to", "openai", "--json"])
    assert json.loads(capsys.readouterr().out)["read_notes"] == {"tagged_calls_unparsed": 1}
    main(["roundtrip", str(src), "--to", "sharegpt"])
    captured = capsys.readouterr()
    assert "1\tread-note\ttagged_calls_unparsed" in captured.out
    assert "read note: tagged_calls_unparsed" in captured.err


@pytest.mark.parametrize(
    "turns, broken_at",
    [
        ([Turn("user", [Text("a")]), Turn("user", [Text("b")]), Turn("assistant", [Text("c")])], 1),
        (
            [
                Turn("user", [Text("a")]),
                Turn("assistant", [ToolCall("f", {}), ToolCall("g", {})]),
                Turn("tool", [ToolResult([Text("1")]), ToolResult([Text("2")])]),
            ],
            2,
        ),
        ([Turn("user", [Text("a")]), Turn("assistant", [Text("b"), ToolCall("f", {})])], 2),
        ([Turn("assistant", [Text("hi")])], 0),
    ],
)
def test_order_rule_breaks_are_reported(turns, broken_at):
    out, losses = SHAREGPT.write(Conversation(turns=turns))
    order = [loss for loss in losses if loss.code == "order_broken"]
    assert [loss.turn for loss in order] == [broken_at]
    assert SHAREGPT.read(out) == SHAREGPT.fit(Conversation(turns=turns), losses)


def test_alternating_conversation_has_no_order_loss():
    conv = SHAREGPT.read(records("sharegpt.jsonl")[1])
    assert SHAREGPT.write(conv)[1] == []
