"""OpenAI chat messages, as used by chat completions and fine-tuning JSONL.

A record is ``{"messages": [...], "tools": [...]}``. Roles are ``system`` (``developer`` is
read as ``system``), ``user``, ``assistant`` with ``tool_calls`` or the older single
``function_call``, ``tool`` with ``tool_call_id``, and the older ``function`` role with a
``name``. Content is a string or a list of typed parts (``text``, ``image_url``).
"""

from __future__ import annotations

import json
from typing import Any

from .. import steps as S
from ..model import Conversation, Image, Text, ToolCall, ToolResult, ToolSpec, Turn
from .base import Format, FormatError, expect, extras

_MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "function_call", "tool_call_id"}


def _text_parts(content: Any, where: str, images: bool) -> list[Text | Image]:
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content)]
    expect(isinstance(content, list), f"{where}: content must be a string, a list or null")
    parts: list[Text | Image] = []
    for item in content:
        expect(isinstance(item, dict), f"{where}: content part must be an object")
        kind = item.get("type")
        if kind == "text":
            expect(isinstance(item.get("text"), str), f"{where}: text part without text")
            parts.append(Text(item["text"], extras(item, {"type", "text"})))
        elif kind == "image_url" and images:
            spec = item.get("image_url")
            if isinstance(spec, str):
                spec = {"url": spec}
            expect(
                isinstance(spec, dict) and isinstance(spec.get("url"), str),
                f"{where}: bad image_url",
            )
            expect(not extras(spec, {"url", "detail"}), f"{where}: unknown image_url field")
            parts.append(
                Image(spec["url"], spec.get("detail"), extras(item, {"type", "image_url"}))
            )
        else:
            raise FormatError(f"{where}: unsupported content part {kind!r}")
    return parts


def _call(item: Any, where: str, legacy: bool) -> ToolCall:
    expect(isinstance(item, dict), f"{where}: tool call must be an object")
    fn = item if legacy else item.get("function")
    if not legacy:
        expect(item.get("type", "function") == "function", f"{where}: unsupported tool call type")
    expect(isinstance(fn, dict) and isinstance(fn.get("name"), str), f"{where}: call without name")
    args = fn.get("arguments", "")
    if isinstance(args, dict):
        arguments: dict[str, Any] | str = args
    else:
        expect(isinstance(args, str), f"{where}: arguments must be a string")
        arguments = args
    if legacy:
        return ToolCall(fn["name"], arguments, None, extras(fn, {"name", "arguments"}))
    expect(not extras(fn, {"name", "arguments"}), f"{where}: unknown function field")
    meta = extras(item, {"id", "type", "function"})
    call_id = item.get("id")
    expect(call_id is None or isinstance(call_id, str), f"{where}: id must be a string")
    return ToolCall(fn["name"], arguments, call_id, meta)


class OpenAI(Format):
    name = "openai"
    steps = (
        S.drop_meta(
            "tool_turn",
            "result_image",
            reserved={
                "conv": {"messages", "tools"},
                "turn": _MESSAGE_KEYS,
                "tool_result": _MESSAGE_KEYS,
                "text": {"type", "text"},
                "system_text": {"type", "text"},
                "result_text": {"type", "text"},
                "image": {"type", "image_url"},
                "tool_call": {"id", "type", "function", "name", "arguments"},
                "tool_spec": {"name", "description", "parameters"},
            },
        ),
        S.images(in_turns=False, in_results=True),
        S.text_before_calls,
        S.assign_tool_ids(allow_legacy=True),
        S.result_names_by_id,
        S.drop_error_flag,
    )

    def decode(self, record: Any) -> Conversation:
        expect(isinstance(record, dict), "record must be an object")
        messages = record.get("messages")
        expect(isinstance(messages, list), "record has no messages list")
        conv = Conversation(meta=extras(record, {"messages", "tools"}))
        tools = record.get("tools") or []
        if isinstance(tools, str):  # some dataset exports store the list as a JSON string
            try:
                tools = json.loads(tools)
            except ValueError as exc:
                raise FormatError("tools field is not JSON") from exc
        expect(isinstance(tools, list), "tools must be a list")
        for spec in tools:
            expect(isinstance(spec, dict), "tool declaration must be an object")
            expect(spec.get("type", "function") == "function", "unsupported tool type")
            expect(not extras(spec, {"type", "function"}), "unknown tool declaration field")
            fn = spec.get("function")
            expect(isinstance(fn, dict) and isinstance(fn.get("name"), str), "tool without name")
            conv.tools.append(
                ToolSpec(
                    fn["name"],
                    fn.get("description"),
                    fn.get("parameters"),
                    extras(fn, {"name", "description", "parameters"}),
                )
            )
        for n, msg in enumerate(messages):
            where = f"message {n}"
            expect(isinstance(msg, dict), f"{where}: not an object")
            role = msg.get("role")
            name = msg.get("name")
            meta = extras(msg, _MESSAGE_KEYS)
            if role in ("system", "developer"):
                parts = _text_parts(msg.get("content"), where, images=False)
                conv.turns.append(Turn("system", list(parts), name, meta))
            elif role == "user":
                parts = _text_parts(msg.get("content"), where, images=True)
                conv.turns.append(Turn("user", list(parts), name, meta))
            elif role == "assistant":
                parts = list(_text_parts(msg.get("content"), where, images=False))
                calls = msg.get("tool_calls") or []
                expect(isinstance(calls, list), f"{where}: tool_calls must be a list")
                parts.extend(_call(c, where, legacy=False) for c in calls)
                if msg.get("function_call") is not None:
                    parts.append(_call(msg["function_call"], where, legacy=True))
                conv.turns.append(Turn("assistant", parts, name, meta))
            elif role == "tool":
                call_id = msg.get("tool_call_id")
                expect(
                    isinstance(call_id, str) or (call_id is None and isinstance(name, str)),
                    f"{where}: tool message without tool_call_id or name",
                )
                content = _text_parts(msg.get("content"), where, images=False)
                conv.turns.append(Turn("tool", [ToolResult(content, call_id, name, meta=meta)]))
            elif role == "function":
                expect(isinstance(name, str), f"{where}: function message without name")
                content = _text_parts(msg.get("content"), where, images=False)
                conv.turns.append(Turn("tool", [ToolResult(content, None, name, meta=meta)]))
            else:
                raise FormatError(f"{where}: unknown role {role!r}")
        return conv

    def encode(self, conv: Conversation) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if conv.system:
            messages.append({"role": "system", "content": _content(conv.system)})
        for turn in conv.turns:
            if turn.role == "tool":
                for res in turn.parts:
                    assert isinstance(res, ToolResult)
                    if res.id is not None:
                        msg = {"role": "tool", "tool_call_id": res.id}
                    else:
                        msg = {"role": "function", "name": res.name}
                    msg["content"] = _content(res.content)
                    messages.append({**res.meta, **msg})
                continue
            msg: dict[str, Any] = {"role": turn.role}
            if turn.name is not None:
                msg["name"] = turn.name
            body = [p for p in turn.parts if not isinstance(p, ToolCall)]
            calls = [p for p in turn.parts if isinstance(p, ToolCall)]
            msg["content"] = _content(body) if body or not calls else None
            if calls and calls[0].id is None:
                (call,) = calls
                msg["function_call"] = {
                    **call.meta,
                    "name": call.name,
                    "arguments": _arguments(call.arguments),
                }
            elif calls:
                msg["tool_calls"] = [_encode_call(c) for c in calls]
            messages.append({**turn.meta, **msg})
        record: dict[str, Any] = {"messages": messages}
        if conv.tools:
            record["tools"] = [
                {"type": "function", "function": _encode_spec(spec)} for spec in conv.tools
            ]
        return {**conv.meta, **record}


def _content(parts: list) -> str | list[dict[str, Any]]:
    if not parts:
        return ""
    if len(parts) == 1 and isinstance(parts[0], Text) and not parts[0].meta:
        return parts[0].text
    out: list[dict[str, Any]] = []
    for p in parts:
        if isinstance(p, Text):
            out.append({**p.meta, "type": "text", "text": p.text})
        else:
            assert isinstance(p, Image)
            spec = {"url": p.url}
            if p.detail is not None:
                spec["detail"] = p.detail
            out.append({**p.meta, "type": "image_url", "image_url": spec})
    return out


def _arguments(arguments: dict[str, Any] | str) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _encode_call(call: ToolCall) -> dict[str, Any]:
    fn = {"name": call.name, "arguments": _arguments(call.arguments)}
    return {**call.meta, "id": call.id, "type": "function", "function": fn}


def _encode_spec(spec: ToolSpec) -> dict[str, Any]:
    out: dict[str, Any] = {**spec.meta, "name": spec.name}
    if spec.description is not None:
        out["description"] = spec.description
    if spec.parameters is not None:
        out["parameters"] = spec.parameters
    return out
