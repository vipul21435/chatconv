"""ShareGPT: ``{"conversations": [{"from": ..., "value": ...}]}``.

Speakers are ``system``, ``human`` (or ``user``) and ``gpt`` (or ``assistant``, ``chatgpt``,
``bard``, ``bing``, ``model``). The tool variant adds ``function_call`` entries whose value is a
JSON object ``{"name", "arguments"}`` (or a list of them), ``observation`` entries holding a
tool's output, and a top-level ``tools`` field with a JSON list of declarations.

A second tool style is read as well: ``gpt`` values made only of
``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` blocks are read as calls, and ``tool``
entries as results (their text is kept as it is, tags included). The writer always uses the
``function_call`` / ``observation`` style. A top-level
``system`` string is read as the system prompt. Values are plain strings, so there is no place
for images, ids, author names or the tool name of a result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import steps as S
from ..model import Conversation, Text, ToolCall, ToolResult, ToolSpec, Turn
from .base import Format, FormatError, expect, extras

_USER = {"human", "user"}
_ASSISTANT = {"gpt", "assistant", "chatgpt", "bard", "bing", "model"}
_RECORD_KEYS = {"conversations", "tools", "system"}
_ENTRY_KEYS = {"from", "value"}
_SPEC_KEYS = {"name", "description", "parameters"}
_TAGGED_CALL = re.compile(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", re.S)


def _tagged_calls(value: str) -> list[ToolCall] | None:
    """Calls from a value made only of ``<tool_call>`` blocks holding JSON, else ``None``.

    This is the tag style of several public function-calling sets in ShareGPT form.
    """
    pos, calls = 0, []
    for m in _TAGGED_CALL.finditer(value):
        if m.start() != pos:
            return None
        pos = m.end()
        try:
            item = json.loads(m.group(1))
        except ValueError:
            return None
        if not (isinstance(item, dict) and isinstance(item.get("name"), str)):
            return None
        if extras(item, {"name", "arguments"}):
            return None
        args = item.get("arguments", {})
        if not isinstance(args, (dict, str)):
            return None
        calls.append(ToolCall(item["name"], args))
    return calls if calls and pos == len(value) else None


def _calls(value: str, where: str) -> list[ToolCall]:
    try:
        data = json.loads(value)
    except ValueError as exc:
        raise FormatError(f"{where}: function_call value is not JSON") from exc
    items = data if isinstance(data, list) else [data]
    calls = []
    for item in items:
        expect(
            isinstance(item, dict) and isinstance(item.get("name"), str),
            f"{where}: function_call without a name",
        )
        expect(not extras(item, {"name", "arguments"}), f"{where}: unknown function_call field")
        args = item.get("arguments", {})
        if not isinstance(args, (dict, str)):
            args = json.dumps(args, ensure_ascii=False)
        calls.append(ToolCall(item["name"], args))
    return calls


def _specs(raw: Any) -> list[ToolSpec]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise FormatError("tools field is not JSON") from exc
    expect(isinstance(raw, list), "tools must be a list")
    specs = []
    for item in raw:
        if isinstance(item, dict) and item.get("type") == "function" and "function" in item:
            item = item["function"]  # OpenAI-style declarations, as some sets store them
        expect(isinstance(item, dict) and isinstance(item.get("name"), str), "tool without name")
        specs.append(
            ToolSpec(
                item["name"],
                item.get("description"),
                item.get("parameters"),
                extras(item, _SPEC_KEYS),
            )
        )
    return specs


class ShareGPT(Format):
    name = "sharegpt"
    steps = (
        S.drop_meta(
            "system_text",
            "tool_turn",
            "text",
            "image",
            "tool_call",
            "result_text",
            "result_image",
            reserved={
                "conv": _RECORD_KEYS,
                "turn": _ENTRY_KEYS,
                "tool_result": _ENTRY_KEYS,
                "tool_spec": _SPEC_KEYS,
            },
        ),
        S.images(),
        S.names,
        S.split_assistant,
        S.join_text("\n"),
        S.drop_tool_ids,
        S.drop_result_names,
        S.drop_error_flag,
    )

    def decode(self, record: Any) -> Conversation:
        expect(isinstance(record, dict), "record must be an object")
        entries = record.get("conversations")
        expect(isinstance(entries, list), "record has no conversations list")
        conv = Conversation(meta=extras(record, _RECORD_KEYS), tools=_specs(record.get("tools")))
        system = record.get("system")
        if system is not None:
            expect(isinstance(system, str), "system must be a string")
            conv.system.append(Text(system))
        for n, entry in enumerate(entries):
            where = f"entry {n}"
            expect(isinstance(entry, dict), f"{where}: not an object")
            speaker, value = entry.get("from"), entry.get("value")
            expect(isinstance(speaker, str), f"{where}: from must be a string")
            expect(isinstance(value, str), f"{where}: value must be a string")
            meta = extras(entry, _ENTRY_KEYS)
            if speaker == "system":
                conv.turns.append(Turn("system", [Text(value)], meta=meta))
            elif speaker in _USER:
                conv.turns.append(Turn("user", [Text(value)], meta=meta))
            elif speaker in _ASSISTANT:
                tagged = _tagged_calls(value)
                parts = tagged if tagged is not None else [Text(value)]
                conv.turns.append(Turn("assistant", list(parts), meta=meta))
            elif speaker == "function_call":
                conv.turns.append(Turn("assistant", list(_calls(value, where)), meta=meta))
            elif speaker in ("observation", "tool"):
                conv.turns.append(Turn("tool", [ToolResult([Text(value)], meta=meta)]))
            else:
                raise FormatError(f"{where}: unknown speaker {speaker!r}")
        return conv

    def encode(self, conv: Conversation) -> dict[str, Any]:
        entries: list[dict[str, Any]] = [{"from": "system", "value": t.text} for t in conv.system]
        for turn in conv.turns:
            if turn.role == "tool":
                for res in turn.parts:
                    assert isinstance(res, ToolResult)
                    entries.append({**res.meta, "from": "observation", "value": _text(res.content)})
                continue
            calls = [p for p in turn.parts if isinstance(p, ToolCall)]
            if calls:
                items = [{"name": c.name, "arguments": c.arguments} for c in calls]
                value = json.dumps(items[0] if len(items) == 1 else items, ensure_ascii=False)
                entries.append({**turn.meta, "from": "function_call", "value": value})
                continue
            speaker = {"system": "system", "user": "human", "assistant": "gpt"}[turn.role]
            entries.append({**turn.meta, "from": speaker, "value": _text(turn.parts)})
        record: dict[str, Any] = {"conversations": entries}
        if conv.tools:
            specs = []
            for spec in conv.tools:
                item: dict[str, Any] = {**spec.meta, "name": spec.name}
                if spec.description is not None:
                    item["description"] = spec.description
                if spec.parameters is not None:
                    item["parameters"] = spec.parameters
                specs.append(item)
            record["tools"] = json.dumps(specs, ensure_ascii=False)
        return {**conv.meta, **record}


def _text(parts: list) -> str:
    texts = [p.text for p in parts if isinstance(p, Text)]
    assert len(texts) == len(parts) and len(texts) <= 1, "unfitted parts"
    return texts[0] if texts else ""
