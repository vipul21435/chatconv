"""Rewrites shared by the writers. Each returns one loss per change and changes nothing else."""

from __future__ import annotations

import json
from itertools import count

from .losses import Loss, Policy, Step, step
from .model import Conversation, Image, Text, ToolCall, ToolResult, Turn, tool_call_names


@step("system turns", "system_turns_dropped", "system_turns_as_user")
def system_turns(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with no system messages after the start."""
    losses: list[Loss] = []
    kept: list[Turn] = []
    for i, turn in enumerate(conv.turns):
        if turn.role != "system":
            kept.append(turn)
        elif policy.system_turns == "drop":
            losses.append(Loss("system_turns_dropped", i))
        else:
            turn.role = "user"
            losses.append(Loss("system_turns_as_user", i))
            kept.append(turn)
    conv.turns = kept
    return losses


def images(in_turns: bool = True, in_results: bool = True) -> Step:
    """For targets with no image parts in turns, in tool results, or both."""

    @step("images", "images_dropped", "images_as_placeholder")
    def run(conv: Conversation, policy: Policy) -> list[Loss]:
        losses: list[Loss] = []
        code = "images_dropped" if policy.images == "drop" else "images_as_placeholder"

        def rewrite(parts: list) -> tuple[list, int]:
            out, n = [], 0
            for p in parts:
                if isinstance(p, Image):
                    n += 1
                    if policy.images == "placeholder":
                        out.append(Text(policy.image_placeholder))
                else:
                    out.append(p)
            return out, n

        for i, turn in enumerate(conv.turns):
            n = 0
            if in_turns:
                turn.parts, n = rewrite(turn.parts)
            if in_results:
                for part in turn.parts:
                    if isinstance(part, ToolResult):
                        part.content, m = rewrite(part.content)
                        n += m
            if n:
                losses.append(Loss(code, i, f"{n} image(s)"))
        return losses

    return run


@step("names", "names_dropped")
def names(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with no author name on a turn."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        if turn.name is not None:
            losses.append(Loss("names_dropped", i, turn.name))
            turn.name = None
    return losses


@step("split assistant turns", "turn_split")
def split_assistant(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets that hold either text or tool calls in one assistant turn, not both.

    The text goes first, then the calls; turn metadata stays on the text turn.
    """
    losses: list[Loss] = []
    turns: list[Turn] = []
    for i, turn in enumerate(conv.turns):
        calls = [p for p in turn.parts if isinstance(p, ToolCall)]
        rest = [p for p in turn.parts if not isinstance(p, ToolCall)]
        if turn.role == "assistant" and calls and rest:
            turns.append(Turn("assistant", rest, turn.name, turn.meta))
            turns.append(Turn("assistant", calls, turn.name))
            losses.append(Loss("turn_split", i))
        else:
            turns.append(turn)
    conv.turns = turns
    return losses


def _join(parts: list, sep: str) -> tuple[list, bool]:
    """Join all text parts into one; report whether anything changed."""
    texts = [p for p in parts if isinstance(p, Text)]
    if len(texts) <= 1:
        return parts, False
    joined = Text(sep.join(t.text for t in texts)) if texts else None
    out = [p for p in parts if not isinstance(p, Text)]
    if joined is not None and joined.text:
        out.insert(0, joined)
    return out, True


def join_text(sep: str = "\n", in_turns: bool = True) -> Step:
    """For targets whose message content is one string.

    Run it after the metadata step: the joined part carries no metadata.
    """

    @step("join text", "text_joined")
    def run(conv: Conversation, policy: Policy) -> list[Loss]:
        losses: list[Loss] = []
        for i, turn in enumerate(conv.turns):
            if turn.role == "tool":
                for part in turn.parts:
                    if isinstance(part, ToolResult):
                        part.content, changed = _join(part.content, sep)
                        if changed:
                            losses.append(Loss("text_joined", i, "tool result"))
                continue
            if not in_turns:
                continue
            turn.parts, changed = _join(turn.parts, sep)
            if changed:
                losses.append(Loss("text_joined", i))
        return losses

    return run


@step("drop tool ids", "tool_ids_dropped")
def drop_tool_ids(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets that do not link results to calls by id."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        n = 0
        for part in turn.parts:
            if isinstance(part, (ToolCall, ToolResult)) and part.id is not None:
                part.id = None
                n += 1
        if n:
            losses.append(Loss("tool_ids_dropped", i, f"{n} id(s)"))
    return losses


@step("drop tool result names", "tool_result_names_dropped")
def drop_result_names(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets whose tool results carry no tool name."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        for part in turn.parts:
            if isinstance(part, ToolResult) and part.name is not None:
                losses.append(Loss("tool_result_names_dropped", i, part.name))
                part.name = None
    return losses


@step("result names follow ids", "tool_result_names_dropped")
def result_names_by_id(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets that know a result's tool only through the id of its call.

    A result with an id gets the name of the call with that id (or no name when no call has
    that id); a different name it carried is lost.
    """
    names_by_id = tool_call_names(conv)
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        for part in turn.parts:
            if isinstance(part, ToolResult) and part.id is not None:
                resolved = names_by_id.get(part.id)
                if part.name != resolved:
                    losses.append(Loss("tool_result_names_dropped", i, str(part.name)))
                    part.name = resolved
    return losses


@step("drop tool error flag", "tool_error_flag_dropped")
def drop_error_flag(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with no way to mark a tool result as an error."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        for part in turn.parts:
            if isinstance(part, ToolResult) and part.is_error:
                part.is_error = False
                losses.append(Loss("tool_error_flag_dropped", i))
    return losses


def assign_tool_ids(allow_legacy: bool) -> Step:
    """For targets that need an id on every call and result.

    Ids are invented as ``call_<n>``, skipping ids already in use. A result without an id is
    linked to the first unanswered call of the latest assistant turn with the same tool name
    (or any name when the result has none) and takes that call's id and name.

    With ``allow_legacy`` (OpenAI's deprecated ``function_call`` form) a call that was read from
    that form keeps it, without an id, when it is the only call of its turn and every result in
    the following tool turn was read from a ``function`` message naming that tool; such a
    result with no call to link to also keeps its form. Every other call or result without an
    id gets one.
    """

    @step("assign tool ids", "tool_ids_invented")
    def run(conv: Conversation, policy: Policy) -> list[Loss]:
        used = set(tool_call_names(conv))
        for turn in conv.turns:
            for part in turn.parts:
                if isinstance(part, ToolResult) and part.id is not None:
                    used.add(part.id)
        counter = count()

        def fresh() -> str:
            while True:
                candidate = f"call_{next(counter)}"
                if candidate not in used:
                    used.add(candidate)
                    return candidate

        losses: list[Loss] = []
        pending: list[ToolCall] = []
        for i, turn in enumerate(conv.turns):
            if turn.role == "assistant":
                calls = [p for p in turn.parts if isinstance(p, ToolCall)]
                if not calls:
                    continue
                legacy = (
                    allow_legacy
                    and len(calls) == 1
                    and calls[0].id is None
                    and calls[0].legacy
                    and _legacy_results_follow(conv, i, calls[0].name)
                )
                n = 0
                for call in calls:
                    if call.id is None and not legacy:
                        call.id = fresh()
                        n += 1
                if n:
                    losses.append(Loss("tool_ids_invented", i, f"{n} call id(s)"))
                pending = [] if legacy else list(calls)
            elif turn.role == "tool":
                n = 0
                for part in turn.parts:
                    if not isinstance(part, ToolResult):
                        continue
                    if part.id is not None:
                        pending = [c for c in pending if c.id != part.id]
                        continue
                    match = next(
                        (c for c in pending if part.name is None or c.name == part.name), None
                    )
                    if match is not None:
                        pending.remove(match)
                        part.id, part.name = match.id, match.name
                        n += 1
                    elif allow_legacy and part.name is not None and part.legacy:
                        continue
                    else:
                        part.id = fresh()
                        n += 1
                if n:
                    losses.append(Loss("tool_ids_invented", i, f"{n} result id(s)"))
        return losses

    return run


def _legacy_results_follow(conv: Conversation, i: int, name: str) -> bool:
    if i + 1 >= len(conv.turns) or conv.turns[i + 1].role != "tool":
        return True
    results = [p for p in conv.turns[i + 1].parts if isinstance(p, ToolResult)]
    return all(r.id is None and r.name == name and r.legacy for r in results)


META_LEVELS = frozenset(
    {
        "conv",
        "tool_spec",
        "system_text",
        "turn",
        "tool_turn",
        "text",
        "image",
        "tool_call",
        "tool_result",
        "result_text",
        "result_image",
    }
)


def _level(part: object, inside_result: bool) -> str:
    if isinstance(part, Text):
        return "result_text" if inside_result else "text"
    if isinstance(part, Image):
        return "result_image" if inside_result else "image"
    return "tool_call" if isinstance(part, ToolCall) else "tool_result"


def _filter(
    meta: dict, level: str, drop: frozenset[str], reserved: dict[str, set[str]]
) -> tuple[dict, list[str]]:
    """Return the metadata a target keeps at ``level`` and the keys it loses."""
    if not meta:
        return meta, []
    if level in drop:
        return {}, sorted(meta)
    clash = reserved.get(level, set()) & set(meta)
    if not clash:
        return meta, []
    return {k: v for k, v in meta.items() if k not in clash}, sorted(clash)


def drop_meta(*lost: str, reserved: dict[str, set[str]] | None = None) -> Step:
    """Drop metadata a target cannot hold.

    ``lost`` names whole levels (from ``META_LEVELS``) the target has no place for;
    ``reserved`` maps a level to keys the target's own fields use, which an unknown field of
    the same name cannot keep. ``policy.extra == "drop"`` drops every level.
    """
    reserved = reserved or {}
    unknown = (set(lost) | set(reserved)) - META_LEVELS
    if unknown:
        raise ValueError(f"unknown metadata levels {sorted(unknown)}")

    @step("drop extra fields", "extra_dropped")
    def run(conv: Conversation, policy: Policy) -> list[Loss]:
        drop = META_LEVELS if policy.extra == "drop" else frozenset(lost)
        losses: list[Loss] = []

        def fix(obj, level: str, turn: int | None) -> None:
            obj.meta, keys = _filter(obj.meta, level, drop, reserved)
            if keys:
                losses.append(Loss("extra_dropped", turn, f"{level}: {', '.join(keys)}"))

        fix(conv, "conv", None)
        for spec in conv.tools:
            fix(spec, "tool_spec", None)
        for text in conv.system:
            fix(text, "system_text", None)
        for i, turn in enumerate(conv.turns):
            fix(turn, "tool_turn" if turn.role == "tool" else "turn", i)
            for part in turn.parts:
                fix(part, _level(part, False), i)
                if isinstance(part, ToolResult):
                    for sub in part.content:
                        fix(sub, _level(sub, True), i)
        return losses

    return run


@step("text before tool calls", "parts_reordered")
def text_before_calls(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets that write an assistant message's text and its tool calls in separate fields."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        if turn.role != "assistant":
            continue
        ordered = [p for p in turn.parts if not isinstance(p, ToolCall)] + [
            p for p in turn.parts if isinstance(p, ToolCall)
        ]
        if ordered != turn.parts:
            turn.parts = ordered
            losses.append(Loss("parts_reordered", i))
    return losses


@step("fold system prompt", "system_folded", "system_dropped")
def fold_system(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with no system slot: prepend it to the first user turn, or drop it."""
    if not conv.system:
        return []
    texts, conv.system = conv.system, []
    if policy.system == "drop":
        return [Loss("system_dropped", None, f"{len(texts)} part(s)")]
    first = next((t for t in conv.turns if t.role == "user"), None)
    if first is None:
        first = Turn("user", [])
        conv.turns.insert(0, first)
    first.parts[:0] = [Text(t.text) for t in texts]
    return [Loss("system_folded", None, f"{len(texts)} part(s)")]


@step("drop tool declarations", "tools_dropped")
def drop_tools(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with no place for tool declarations."""
    if not conv.tools:
        return []
    names = ", ".join(spec.name for spec in conv.tools)
    conv.tools = []
    return [Loss("tools_dropped", None, names)]


def call_text(call: ToolCall) -> str:
    """The text a tool call becomes in a target without tool calls."""
    return json.dumps({"name": call.name, "arguments": call.arguments}, ensure_ascii=False)


def tools_as_text(conv: Conversation, policy: Policy, results: bool = True) -> list[Loss]:
    """For targets with no tool calls (and, with ``results``, no tool turns either).

    With ``policy.tools == "text"`` each call becomes a JSON text part in its place and each
    tool turn becomes a user turn holding the results' text; with ``drop`` both go.
    """
    losses: list[Loss] = []
    turns: list[Turn] = []
    for i, turn in enumerate(conv.turns):
        if turn.role == "tool" and results:
            if policy.tools == "drop":
                losses.append(Loss("tool_results_dropped", i, f"{len(turn.parts)} result(s)"))
                continue
            parts: list = []
            for res in turn.parts:
                assert isinstance(res, ToolResult)
                parts.extend(res.content)
            turns.append(Turn("user", parts, None, turn.meta))
            losses.append(Loss("tool_results_as_text", i, f"{len(turn.parts)} result(s)"))
            continue
        calls = [p for p in turn.parts if isinstance(p, ToolCall)]
        if calls and turn.role == "assistant":
            if policy.tools == "drop":
                turn.parts = [p for p in turn.parts if not isinstance(p, ToolCall)]
                losses.append(Loss("tool_calls_dropped", i, f"{len(calls)} call(s)"))
            else:
                turn.parts = [
                    Text(call_text(p)) if isinstance(p, ToolCall) else p for p in turn.parts
                ]
                losses.append(Loss("tool_calls_as_text", i, f"{len(calls)} call(s)"))
        turns.append(turn)
    conv.turns = turns
    return losses


def alternate(first: str = "user", last: str | None = None) -> Step:
    """For targets that need user and assistant turns to alternate.

    Consecutive turns with the same role merge (parts concatenated, the first turn's name and
    metadata kept); an empty turn is added where the order needs one: before an opening
    assistant turn when ``first`` is ``user``, and at the end when ``last`` names a role.
    Run it after tool and system turns have been removed.
    """

    @step("alternate roles", "turns_merged", "turns_added")
    def run(conv: Conversation, policy: Policy) -> list[Loss]:
        losses: list[Loss] = []
        turns: list[Turn] = []
        for i, turn in enumerate(conv.turns):
            if turns and turns[-1].role == turn.role:
                turns[-1].parts.extend(turn.parts)
                for key, value in turn.meta.items():
                    turns[-1].meta.setdefault(key, value)
                losses.append(Loss("turns_merged", i, turn.role))
                continue
            turns.append(turn)
        if first == "user" and (not turns or turns[0].role != "user"):
            turns.insert(0, Turn("user", []))
            losses.append(Loss("turns_added", 0, "user"))
        if last is not None and turns[-1].role != last:
            turns.append(Turn(last, []))
            losses.append(Loss("turns_added", len(turns) - 1, last))
        conv.turns = turns
        return losses

    return run


@step("wrap arguments", "arguments_wrapped")
def wrap_arguments(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets whose call arguments must be a JSON object: raw text goes under ``_raw``."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        for part in turn.parts:
            if isinstance(part, ToolCall) and isinstance(part.arguments, str):
                part.arguments = {"_raw": part.arguments}
                losses.append(Loss("arguments_wrapped", i, part.name))
    return losses


@step("image detail", "image_detail_dropped")
def drop_image_detail(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with no image detail setting."""
    losses: list[Loss] = []
    for i, turn in enumerate(conv.turns):
        parts = list(turn.parts)
        for part in turn.parts:
            if isinstance(part, ToolResult):
                parts.extend(part.content)
        n = 0
        for part in parts:
            if isinstance(part, Image) and part.detail is not None:
                part.detail = None
                n += 1
        if n:
            losses.append(Loss("image_detail_dropped", i, f"{n} image(s)"))
    return losses


@step("infer result names", "tool_result_names_inferred")
def infer_result_names(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets that need a tool name on every result.

    A result with neither id nor name takes the name of the first unanswered call of the
    latest assistant turn, or ``unknown`` when there is none.
    """
    losses: list[Loss] = []
    pending: list[ToolCall] = []
    for i, turn in enumerate(conv.turns):
        if turn.role == "assistant":
            calls = [p for p in turn.parts if isinstance(p, ToolCall)]
            if calls:
                pending = list(calls)
            continue
        for part in turn.parts:
            if not isinstance(part, ToolResult):
                continue
            if part.id is not None:
                match = next((c for c in pending if c.id == part.id), None)
            elif part.name is not None:
                match = next((c for c in pending if c.name == part.name), None)
            else:
                match = pending[0] if pending else None
            if part.name is None:
                part.name = match.name if match is not None else "unknown"
                losses.append(Loss("tool_result_names_inferred", i, part.name))
            pending = [c for c in pending if c is not match]
    return losses


tools_to_text = step(
    "tools as text",
    "tool_calls_as_text",
    "tool_calls_dropped",
    "tool_results_as_text",
    "tool_results_dropped",
)(tools_as_text)
"""Calls and tool turns become text (or go), per ``policy.tools``."""

calls_to_text = step("calls as text", "tool_calls_as_text", "tool_calls_dropped")(
    lambda conv, policy: tools_as_text(conv, policy, results=False)
)
"""Calls become text (or go); tool turns stay."""


@step("join system prompt", "text_joined")
def join_system(conv: Conversation, policy: Policy) -> list[Loss]:
    """For targets with one system string."""
    if len(conv.system) <= 1:
        return []
    n = len(conv.system)
    conv.system = [Text("\n".join(t.text for t in conv.system))]
    return [Loss("text_joined", None, f"system, {n} parts")]


@step("trim text", "text_trimmed")
def trim_text(conv: Conversation, policy: Policy) -> list[Loss]:
    """For renderers whose template strips each message (Jinja ``trim``, i.e. ``str.strip``)."""
    losses: list[Loss] = []
    for text in conv.system:
        if text.text != text.text.strip():
            text.text = text.text.strip()
            losses.append(Loss("text_trimmed", None, "system"))
    for i, turn in enumerate(conv.turns):
        changed = False
        for part in turn.parts:
            if isinstance(part, Text) and part.text != part.text.strip():
                part.text = part.text.strip()
                changed = True
        if changed:
            losses.append(Loss("text_trimmed", i))
    return losses
