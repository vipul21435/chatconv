"""Anthropic Messages API: a top-level ``system``, ``messages`` of content blocks, ``tools``.

Content is a string or a list of blocks: ``text``, ``image`` (``source`` of type ``base64``
or ``url``), ``tool_use`` (``id``, ``name``, ``input``) in assistant messages and
``tool_result`` (``tool_use_id``, ``content``, ``is_error``) in user messages. Tool results
are read into a tool turn; other blocks of the same user message follow as a user turn.
Images are written as base64 sources when they are ``data:...;base64,`` URIs, else as URL
sources. There is no author name and no image detail setting.
"""

from __future__ import annotations

import re
from typing import Any

from .. import steps as S
from ..model import Conversation, Image, Text, ToolCall, ToolResult, ToolSpec, Turn
from .base import Format, FormatError, expect, extras

_DATA_URI = re.compile(r"^data:([^;,]+);base64,(.*)$", re.S)
_TEXT_KEYS = {"type", "text"}
_IMAGE_KEYS = {"type", "source"}
_USE_KEYS = {"type", "id", "name", "input"}
_RESULT_KEYS = {"type", "tool_use_id", "content", "is_error"}
_MSG_KEYS = {"role", "content"}
_SPEC_KEYS = {"name", "description", "input_schema"}


def _image(block: dict[str, Any], where: str) -> Image:
    src = block.get("source")
    expect(isinstance(src, dict), f"{where}: image without source")
    if src.get("type") == "base64":
        expect(
            isinstance(src.get("data"), str) and isinstance(src.get("media_type"), str),
            f"{where}: base64 image needs data and media_type",
        )
        expect(not extras(src, {"type", "data", "media_type"}), f"{where}: unknown source field")
        url = f"data:{src['media_type']};base64,{src['data']}"
    elif src.get("type") == "url":
        expect(isinstance(src.get("url"), str), f"{where}: url image without url")
        expect(not extras(src, {"type", "url"}), f"{where}: unknown source field")
        url = src["url"]
    else:
        raise FormatError(f"{where}: unsupported image source {src.get('type')!r}")
    return Image(url, None, extras(block, _IMAGE_KEYS))


def _blocks(content: Any, where: str, allowed: set[str]) -> list[Any]:
    if isinstance(content, str):
        return [Text(content)]
    if content is None:
        return []
    expect(isinstance(content, list), f"{where}: content must be a string or a list")
    out: list[Any] = []
    for block in content:
        expect(isinstance(block, dict), f"{where}: block must be an object")
        kind = block.get("type")
        if not isinstance(kind, str) or kind not in allowed:
            raise FormatError(f"{where}: unsupported block {kind!r}")
        if kind == "text":
            expect(isinstance(block.get("text"), str), f"{where}: text block without text")
            out.append(Text(block["text"], extras(block, _TEXT_KEYS)))
        elif kind == "image":
            out.append(_image(block, where))
        elif kind == "tool_use":
            expect(isinstance(block.get("name"), str), f"{where}: tool_use without name")
            expect(isinstance(block.get("id"), str), f"{where}: tool_use without id")
            args = block.get("input", {})
            expect(isinstance(args, dict), f"{where}: tool_use input must be an object")
            out.append(ToolCall(block["name"], args, block["id"], extras(block, _USE_KEYS)))
        else:  # tool_result
            use_id = block.get("tool_use_id")
            expect(isinstance(use_id, str), f"{where}: tool_result without tool_use_id")
            inner = _blocks(block.get("content"), where, {"text", "image"})
            is_error = block.get("is_error", False)
            expect(isinstance(is_error, bool), f"{where}: is_error must be a boolean")
            out.append(ToolResult(inner, use_id, None, is_error, extras(block, _RESULT_KEYS)))
    return out


class Anthropic(Format):
    name = "anthropic"
    steps = (
        S.drop_meta(
            reserved={
                "conv": {"system", "messages", "tools"},
                "turn": _MSG_KEYS,
                "tool_turn": _MSG_KEYS,
                "system_text": _TEXT_KEYS,
                "text": _TEXT_KEYS,
                "result_text": _TEXT_KEYS,
                "image": _IMAGE_KEYS,
                "result_image": _IMAGE_KEYS,
                "tool_call": _USE_KEYS,
                "tool_result": _RESULT_KEYS,
                "tool_spec": _SPEC_KEYS,
            },
        ),
        S.names,
        S.drop_image_detail,
        S.wrap_arguments,
        S.assign_tool_ids(allow_legacy=False),
        S.result_names_by_id,
    )

    def decode(self, record: Any) -> Conversation:
        expect(isinstance(record, dict), "record must be an object")
        messages = record.get("messages")
        expect(isinstance(messages, list), "record has no messages list")
        conv = Conversation(meta=extras(record, {"system", "messages", "tools"}))
        system = record.get("system")
        if system is not None:
            conv.system = _blocks(system, "system", {"text"})
        tools = record.get("tools") or []
        expect(isinstance(tools, list), "tools must be a list")
        for spec in tools:
            expect(
                isinstance(spec, dict) and isinstance(spec.get("name"), str), "tool without name"
            )
            conv.tools.append(
                ToolSpec(
                    spec["name"],
                    spec.get("description"),
                    spec.get("input_schema"),
                    extras(spec, _SPEC_KEYS),
                )
            )
        for n, msg in enumerate(messages):
            where = f"message {n}"
            expect(isinstance(msg, dict), f"{where}: not an object")
            role, meta = msg.get("role"), extras(msg, _MSG_KEYS)
            if role == "assistant":
                parts = _blocks(msg.get("content"), where, {"text", "tool_use"})
                conv.turns.append(Turn("assistant", parts, None, meta))
            elif role == "system":
                conv.turns.append(
                    Turn("system", _blocks(msg.get("content"), where, {"text"}), None, meta)
                )
            elif role == "user":
                parts = _blocks(msg.get("content"), where, {"text", "image", "tool_result"})
                results = [p for p in parts if isinstance(p, ToolResult)]
                rest = [p for p in parts if not isinstance(p, ToolResult)]
                if results:
                    conv.turns.append(Turn("tool", results, None, meta))
                    if rest:
                        conv.turns.append(Turn("user", rest))
                else:
                    conv.turns.append(Turn("user", rest, None, meta))
            else:
                raise FormatError(f"{where}: unknown role {role!r}")
        return conv

    def encode(self, conv: Conversation) -> dict[str, Any]:
        record: dict[str, Any] = {}
        if conv.system:
            record["system"] = _content(conv.system)
        messages = []
        for turn in conv.turns:
            role = "user" if turn.role == "tool" else turn.role
            messages.append({**turn.meta, "role": role, "content": _content(turn.parts)})
        record["messages"] = messages
        if conv.tools:
            specs = []
            for spec in conv.tools:
                item: dict[str, Any] = {**spec.meta, "name": spec.name}
                if spec.description is not None:
                    item["description"] = spec.description
                if spec.parameters is not None:
                    item["input_schema"] = spec.parameters
                specs.append(item)
            record["tools"] = specs
        return {**conv.meta, **record}


def _content(parts: list) -> str | list[dict[str, Any]]:
    if not parts:
        return ""
    if len(parts) == 1 and isinstance(parts[0], Text) and not parts[0].meta:
        return parts[0].text
    return [_block(p) for p in parts]


def _block(part: Any) -> dict[str, Any]:
    if isinstance(part, Text):
        return {**part.meta, "type": "text", "text": part.text}
    if isinstance(part, Image):
        m = _DATA_URI.match(part.url)
        if m:
            source = {"type": "base64", "media_type": m.group(1), "data": m.group(2)}
        else:
            source = {"type": "url", "url": part.url}
        return {**part.meta, "type": "image", "source": source}
    if isinstance(part, ToolCall):
        return {
            **part.meta,
            "type": "tool_use",
            "id": part.id,
            "name": part.name,
            "input": part.arguments,
        }
    assert isinstance(part, ToolResult)
    out = {**part.meta, "type": "tool_result", "tool_use_id": part.id}
    if part.content:
        out["content"] = _content(part.content)
    if part.is_error:
        out["is_error"] = True
    return out
