"""Google Gemini ``contents``: ``{"role": "user" | "model", "parts": [...]}``.

Parts hold ``text``, ``inlineData`` (``mimeType``, ``data``), ``fileData`` (``mimeType``,
``fileUri``), ``functionCall`` (``name``, ``args``, optional ``id``) or ``functionResponse``
(``name``, ``response``, optional ``id``). The system prompt is ``systemInstruction`` (also
read from ``system_instruction``) and declarations are ``tools[].functionDeclarations``.

``response`` is a JSON object. chatconv writes a result's text as ``{"result": text}`` and
reads ``{"result": <string>}`` back as that text; any other object is read as its JSON text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import steps as S
from ..model import Conversation, Image, Text, ToolCall, ToolResult, ToolSpec, Turn
from .base import Format, FormatError, expect, extras

_DATA_URI = re.compile(r"^data:([^;,]+);base64,(.*)$", re.S)
_PART_KEYS = {"text", "inlineData", "fileData", "functionCall", "functionResponse"}
_CONTENT_KEYS = {"role", "parts"}
_SPEC_KEYS = {"name", "description", "parameters"}


def _part(item: Any, where: str) -> Text | Image | ToolCall | ToolResult:
    expect(isinstance(item, dict), f"{where}: part must be an object")
    kinds = [k for k in _PART_KEYS if k in item]
    expect(len(kinds) == 1, f"{where}: part needs exactly one of {sorted(_PART_KEYS)}")
    (kind,) = kinds
    value, meta = item[kind], extras(item, _PART_KEYS)
    if kind == "text":
        expect(isinstance(value, str), f"{where}: text must be a string")
        return Text(value, meta)
    if kind == "inlineData":
        expect(
            isinstance(value, dict) and isinstance(value.get("data"), str),
            f"{where}: bad inlineData",
        )
        mime = value.get("mimeType", "")
        expect(str(mime).startswith("image/"), f"{where}: unsupported inlineData {mime!r}")
        return Image(f"data:{mime};base64,{value['data']}", None, meta)
    if kind == "fileData":
        expect(
            isinstance(value, dict) and isinstance(value.get("fileUri"), str),
            f"{where}: bad fileData",
        )
        if "mimeType" in value:
            meta["mimeType"] = value["mimeType"]
        return Image(value["fileUri"], None, meta)
    expect(
        isinstance(value, dict) and isinstance(value.get("name"), str),
        f"{where}: {kind} without name",
    )
    call_id = value.get("id")
    expect(call_id is None or isinstance(call_id, str), f"{where}: id must be a string")
    if kind == "functionCall":
        args = value.get("args", {})
        expect(isinstance(args, dict), f"{where}: args must be an object")
        expect(not extras(value, {"name", "args", "id"}), f"{where}: unknown functionCall field")
        return ToolCall(value["name"], args, call_id, meta)
    response = value.get("response", {})
    expect(isinstance(response, dict), f"{where}: response must be an object")
    expect(
        not extras(value, {"name", "response", "id"}), f"{where}: unknown functionResponse field"
    )
    if set(response) == {"result"} and isinstance(response["result"], str):
        text = response["result"]
    else:
        text = json.dumps(response, ensure_ascii=False)
    return ToolResult([Text(text)], call_id, value["name"], False, meta)


class Gemini(Format):
    name = "gemini"
    steps = (
        S.drop_meta(
            "result_text",
            "result_image",
            reserved={
                "conv": {"contents", "systemInstruction", "system_instruction", "tools"},
                "turn": _CONTENT_KEYS,
                "tool_turn": _CONTENT_KEYS,
                "text": _PART_KEYS,
                "system_text": _PART_KEYS,
                "image": _PART_KEYS,
                "tool_call": _PART_KEYS,
                "tool_result": _PART_KEYS,
                "tool_spec": _SPEC_KEYS,
            },
        ),
        S.system_turns,
        S.names,
        S.drop_image_detail,
        S.images(in_turns=False, in_results=True),
        S.join_text("\n", in_turns=False),
        S.wrap_arguments,
        S.drop_error_flag,
        S.infer_result_names,
    )

    def decode(self, record: Any) -> Conversation:
        expect(isinstance(record, dict), "record must be an object")
        contents = record.get("contents")
        expect(isinstance(contents, list), "record has no contents list")
        known = {"contents", "systemInstruction", "system_instruction", "tools"}
        conv = Conversation(meta=extras(record, known))
        system = record.get("systemInstruction", record.get("system_instruction"))
        if system is not None:
            expect(isinstance(system, dict), "systemInstruction must be an object")
            for item in system.get("parts") or []:
                part = _part(item, "systemInstruction")
                expect(isinstance(part, Text), "systemInstruction holds text only")
                conv.system.append(part)
        for tool in record.get("tools") or []:
            expect(isinstance(tool, dict), "tool must be an object")
            expect(set(tool) <= {"functionDeclarations"}, "unsupported tool kind")
            for fn in tool.get("functionDeclarations") or []:
                expect(
                    isinstance(fn, dict) and isinstance(fn.get("name"), str), "tool without name"
                )
                conv.tools.append(
                    ToolSpec(
                        fn["name"],
                        fn.get("description"),
                        fn.get("parameters"),
                        extras(fn, _SPEC_KEYS),
                    )
                )
        for n, content in enumerate(contents):
            where = f"content {n}"
            expect(isinstance(content, dict), f"{where}: not an object")
            role = content.get("role", "user")
            parts = [_part(p, where) for p in content.get("parts") or []]
            meta = extras(content, _CONTENT_KEYS)
            results = [p for p in parts if isinstance(p, ToolResult)]
            rest = [p for p in parts if not isinstance(p, ToolResult)]
            if role == "model":
                expect(not results, f"{where}: functionResponse in a model turn")
                expect(
                    not any(isinstance(p, Image) for p in rest), f"{where}: image in a model turn"
                )
                conv.turns.append(Turn("assistant", rest, None, meta))
            elif role in ("user", "function"):
                expect(
                    not any(isinstance(p, ToolCall) for p in rest),
                    f"{where}: functionCall in a user turn",
                )
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
            record["systemInstruction"] = {"parts": [_encode(p) for p in conv.system]}
        contents = []
        for turn in conv.turns:
            role = "model" if turn.role == "assistant" else "user"
            contents.append({**turn.meta, "role": role, "parts": [_encode(p) for p in turn.parts]})
        record["contents"] = contents
        if conv.tools:
            decls = []
            for spec in conv.tools:
                item: dict[str, Any] = {**spec.meta, "name": spec.name}
                if spec.description is not None:
                    item["description"] = spec.description
                if spec.parameters is not None:
                    item["parameters"] = spec.parameters
                decls.append(item)
            record["tools"] = [{"functionDeclarations": decls}]
        return {**conv.meta, **record}


def _encode(part: Any) -> dict[str, Any]:
    meta = dict(part.meta)
    if isinstance(part, Text):
        return {**meta, "text": part.text}
    if isinstance(part, Image):
        m = _DATA_URI.match(part.url)
        if m:
            return {**meta, "inlineData": {"mimeType": m.group(1), "data": m.group(2)}}
        data = {"fileUri": part.url}
        if "mimeType" in meta:
            data["mimeType"] = meta.pop("mimeType")
        return {**meta, "fileData": data}
    if isinstance(part, ToolCall):
        call: dict[str, Any] = {"name": part.name, "args": part.arguments}
        if part.id is not None:
            call["id"] = part.id
        return {**meta, "functionCall": call}
    assert isinstance(part, ToolResult)
    text = "".join(p.text for p in part.content if isinstance(p, Text))
    resp: dict[str, Any] = {"name": part.name, "response": {"result": text}}
    if part.id is not None:
        resp["id"] = part.id
    return {**meta, "functionResponse": resp}
