import pytest

from chatconv.losses import LOSS_CODES, Loss, Policy, apply_losses, run_steps, step, summarize
from chatconv.model import (
    Conversation,
    Image,
    InvalidConversation,
    Text,
    ToolCall,
    ToolResult,
    Turn,
    canonicalize,
    parse_arguments,
    validate,
)


def test_canonicalize_hoists_leading_system_and_merges_tool_turns():
    conv = Conversation(
        turns=[
            Turn("system", [Text("be brief")]),
            Turn("user", [Text("weather?"), Text("")]),
            Turn("assistant", [ToolCall("forecast", '{"city": "Pune"}', id="c1")]),
            Turn("tool", [ToolResult([Text("31 C")], id="c1")]),
            Turn("tool", [ToolResult([Text("")], id="c9")]),
        ]
    )
    canonicalize(conv)
    validate(conv)
    assert conv.system == [Text("be brief")]
    assert [t.role for t in conv.turns] == ["user", "assistant", "tool"]
    assert conv.turns[0].parts == [Text("weather?")]
    assert conv.turns[1].parts[0].arguments == {"city": "Pune"}
    first, second = conv.turns[2].parts
    assert first.name == "forecast"  # resolved from the call id
    assert second.name is None and second.content == []


def test_system_turn_with_name_is_not_hoisted():
    conv = canonicalize(Conversation(turns=[Turn("system", [Text("x")], name="ops")]))
    assert conv.system == [] and conv.turns[0].role == "system"


def test_parse_arguments_keeps_non_objects_raw():
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("[1, 2]") == "[1, 2]"
    assert parse_arguments("{broken") == "{broken"


@pytest.mark.parametrize(
    "turn",
    [
        Turn("robot", []),
        Turn("user", [ToolCall("f", {})]),
        Turn("assistant", [Image("https://example.org/a.png")]),
        Turn("assistant", [ToolResult([Text("x")])]),
        Turn("tool", [Text("x")]),
        Turn("tool", []),
        Turn("user", [Text("")]),
    ],
)
def test_validate_rejects(turn):
    with pytest.raises(InvalidConversation):
        validate(Conversation(turns=[turn]))


def test_loss_codes_are_checked():
    with pytest.raises(ValueError):
        Loss("not_a_code")
    assert Loss("names_dropped", 3, "x").to_dict() == {
        "code": "names_dropped",
        "turn": 3,
        "detail": "x",
    }
    assert all(desc for desc in LOSS_CODES.values())


def test_policy_rejects_unknown_choice():
    with pytest.raises(ValueError):
        Policy(images="keep")


@step("drop names", "names_dropped")
def _drop_names(conv, policy):
    losses = []
    for i, turn in enumerate(conv.turns):
        if turn.name is not None:
            turn.name = None
            losses.append(Loss("names_dropped", i))
    return losses


@step("drop images", "images_dropped")
def _drop_images(conv, policy):
    losses = []
    for i, turn in enumerate(conv.turns):
        kept = [p for p in turn.parts if not isinstance(p, Image)]
        if len(kept) != len(turn.parts):
            turn.parts = kept
            losses.append(Loss("images_dropped", i))
    return losses


def test_run_steps_copies_and_apply_losses_replays_only_reported_steps():
    conv = Conversation(turns=[Turn("user", [Text("hi")], name="ana")])
    out, losses = run_steps(conv, [_drop_names, _drop_images], Policy())
    assert conv.turns[0].name == "ana"  # the input is untouched
    assert summarize(losses) == {"names_dropped": 1}
    assert apply_losses(conv, [_drop_names, _drop_images], losses, Policy()) == out
    assert apply_losses(conv, [_drop_names, _drop_images], [], Policy()) == conv


def test_step_cannot_emit_undeclared_code():
    @step("bad", "names_dropped")
    def bad(conv, policy):
        return [Loss("images_dropped")]

    with pytest.raises(AssertionError):
        bad(Conversation(), Policy())
