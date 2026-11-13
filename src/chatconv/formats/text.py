"""Formats that render a conversation as one string, stored as ``{"text": ...}``.

* ``chatml`` reads and writes ``<|im_start|>{role}\\n{content}<|im_end|>`` blocks with the roles
  ``system``, ``user``, ``assistant`` and ``tool`` (one ``tool`` block per tool result).
* ``llama3`` writes the Llama 3 instruct layout
  ``<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>``, with
  each content stripped of surrounding whitespace as the published chat template does.
* ``mistral`` writes the layout of the chat template published with Mistral-7B-Instruct v0.1
  and v0.2: ``<s> [INST] {system}\\n\\n{user} [/INST] {assistant}</s> [INST] {user} [/INST]``.

The last two are one-way: they render for training or inference and are not read back.
Tool calls (and, for the one-way renderers, tool results) become text per ``--tools``.
"""

from __future__ import annotations

import re
from typing import Any

from .. import steps as S
from ..losses import Loss, Policy, step
from ..model import Conversation, Text, ToolResult, Turn
from .base import Format, FormatError, expect, extras

_ALL_BUT_CONV = (
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
)
_BLOCK = re.compile(r"<\|im_start\|>([^\n]*)\n(.*?)<\|im_end\|>", re.S)
_ROLES = {"system", "user", "assistant", "tool"}


def _text(parts: list) -> str:
    return "".join(p.text for p in parts if isinstance(p, Text))


def _check(text: str, tokens: tuple[str, ...]) -> str:
    for token in tokens:
        if token in text:
            raise FormatError(f"text contains the control token {token}")
    return text


class ChatML(Format):
    name = "chatml"
    steps = (
        S.drop_meta(*_ALL_BUT_CONV, reserved={"conv": {"text"}}),
        S.drop_tools,
        S.calls_to_text,
        S.names,
        S.images(),
        S.join_text("\n"),
        S.drop_tool_ids,
        S.drop_result_names,
        S.drop_error_flag,
    )
    tokens = ("<|im_start|>", "<|im_end|>")

    def decode(self, record: Any) -> Conversation:
        expect(
            isinstance(record, dict) and isinstance(record.get("text"), str), "record has no text"
        )
        text = record["text"]
        conv = Conversation(meta=extras(record, {"text"}))
        pos = 0
        for m in _BLOCK.finditer(text):
            expect(not text[pos : m.start()].strip(), f"text outside a block at offset {pos}")
            pos = m.end()
            role, content = m.group(1), m.group(2)
            expect(role in _ROLES, f"unknown role {role!r} at offset {m.start()}")
            parts = [Text(content)]
            if role == "tool":
                conv.turns.append(Turn("tool", [ToolResult(parts)]))
            else:
                conv.turns.append(Turn(role, list(parts)))
        expect(not text[pos:].strip(), f"text outside a block at offset {pos}")
        return conv

    def encode(self, conv: Conversation) -> dict[str, Any]:
        blocks = [("system", t.text) for t in conv.system]
        for turn in conv.turns:
            if turn.role == "tool":
                blocks += [("tool", _text(r.content)) for r in turn.parts]
            else:
                blocks.append((turn.role, _text(turn.parts)))
        rendered = "".join(
            f"<|im_start|>{role}\n{_check(body, self.tokens)}<|im_end|>\n" for role, body in blocks
        )
        return {**conv.meta, "text": rendered}


class Llama3(Format):
    name = "llama3"
    one_way = True
    steps = (
        S.drop_meta(*_ALL_BUT_CONV, reserved={"conv": {"text"}}),
        S.drop_tools,
        S.tools_to_text,
        S.names,
        S.images(),
        S.join_system,
        S.join_text("\n"),
        S.trim_text,
    )
    tokens = ("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")

    def encode(self, conv: Conversation) -> dict[str, Any]:
        blocks = [("system", t.text) for t in conv.system]
        blocks += [(t.role, _text(t.parts)) for t in conv.turns]
        body = "".join(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{_check(text, self.tokens)}<|eot_id|>"
            for role, text in blocks
        )
        return {**conv.meta, "text": "<|begin_of_text|>" + body}


@step("mistral system prompt", "system_folded", "system_dropped")
def _mistral_system(conv: Conversation, policy: Policy) -> list[Loss]:
    """The template puts the system prompt inside the first instruction, or nowhere."""
    if not conv.system:
        return []
    if policy.system == "drop" or not conv.turns:
        conv.system = []
        return [Loss("system_dropped")]
    return [Loss("system_folded", None, "into the first [INST] block")]


class Mistral(Format):
    name = "mistral"
    one_way = True
    steps = (
        S.drop_meta(*_ALL_BUT_CONV, reserved={"conv": {"text"}}),
        S.drop_tools,
        S.tools_to_text,
        S.system_turns,
        S.names,
        S.images(),
        S.join_system,
        _mistral_system,
        S.alternate(first="user", fill_empty=False),
        S.join_text("\n"),
    )
    tokens = ("<s>", "</s>", "[INST]", "[/INST]")

    def encode(self, conv: Conversation) -> dict[str, Any]:
        out = "<s>"
        for k, turn in enumerate(conv.turns):
            content = _check(_text(turn.parts), self.tokens)
            if turn.role == "user":
                if k == 0 and conv.system:
                    content = _check(conv.system[0].text, self.tokens) + "\n\n" + content
                out += f" [INST] {content} [/INST]"
            else:
                out += f" {content}</s>"
        return {**conv.meta, "text": out}
