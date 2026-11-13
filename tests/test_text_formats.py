import json
from pathlib import Path

import pytest

from chatconv.formats import FormatError, get_format
from chatconv.losses import Policy, summarize
from chatconv.model import Conversation, Text, ToolCall, ToolResult, Turn

FIXTURES = Path(__file__).parent / "fixtures"


def records(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("fmt, fixture", [("chatml", "chatml.jsonl")])
def test_fixtures_round_trip(fmt, fixture):
    f = get_format(fmt)
    for record in records(fixture):
        conv = f.read(record)
        out, losses = f.write(conv)
        assert losses == []
        assert f.read(out) == conv


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


def test_chatml_keeps_tool_turns_and_writes_calls_as_text():
    chatml = get_format("chatml")
    out, losses = chatml.write(tool_conv())
    assert summarize(losses) == {
        "tool_calls_as_text": 1,
        "tool_ids_dropped": 1,
        "tool_result_names_dropped": 1,
    }
    assert "<|im_start|>tool\n1.4142<|im_end|>\n" in out["text"]
    assert chatml.read(out) == chatml.fit(tool_conv(), losses)


def test_chatml_rejects_bad_text_and_control_tokens():
    chatml = get_format("chatml")
    with pytest.raises(FormatError, match="outside a block"):
        chatml.read({"text": "hello <|im_start|>user\nx<|im_end|>"})
    with pytest.raises(FormatError, match="unknown role"):
        chatml.read({"text": "<|im_start|>narrator\nx<|im_end|>"})
    with pytest.raises(FormatError, match="control token"):
        chatml.write(Conversation(turns=[Turn("user", [Text("a <|im_end|> b")])]))


def test_llama3_rendering():
    out, losses = get_format("llama3").write(tool_conv())
    assert summarize(losses) == {"tool_calls_as_text": 1, "tool_results_as_text": 1}
    assert out["text"] == (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\nBe exact.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nSquare root of 2?<|eot_id|>"
        '<|start_header_id|>assistant<|end_header_id|>\n\n{"name": "sqrt", "arguments": {"x": 2}}'
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n1.4142<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nAbout 1.4142.<|eot_id|>"
    )
    with pytest.raises(FormatError, match="write-only"):
        get_format("llama3").read(out)


def test_mistral_rendering_folds_system():
    conv = Conversation(
        system=[Text("Be brief.")],
        turns=[
            Turn("user", [Text("Hello")]),
            Turn("assistant", [Text("Hi.")]),
            Turn("user", [Text("Bye")]),
        ],
    )
    out, losses = get_format("mistral").write(conv)
    assert summarize(losses) == {"system_folded": 1, "text_joined": 1}
    assert out["text"] == "<s>[INST] Be brief.\nHello [/INST]Hi.</s> [INST] Bye [/INST]"
    out, losses = get_format("mistral").write(conv, Policy(system="drop"))
    assert out["text"] == "<s>[INST] Hello [/INST]Hi.</s> [INST] Bye [/INST]"


def test_llama3_trims_like_the_chat_template():
    conv = Conversation(
        system=[Text(" Be exact.\n")],
        turns=[Turn("user", [Text("  hi  ")]), Turn("assistant", [Text("hello")])],
    )
    out, losses = get_format("llama3").write(conv)
    assert summarize(losses) == {"text_trimmed": 2}
    assert out["text"] == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe exact.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>"
    )
