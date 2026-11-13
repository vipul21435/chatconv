"""Guess the format of one record from its shape."""

from __future__ import annotations

from typing import Any

_OPENAI_ROLES = {"system", "developer", "tool", "function"}
_OPENAI_KEYS = {"tool_calls", "function_call", "tool_call_id"}
_OPENAI_PARTS = {"image_url", "input_audio", "file", "refusal"}
_ANTHROPIC_BLOCKS = {
    "tool_use",
    "tool_result",
    "image",
    "document",
    "thinking",
    "redacted_thinking",
}


def _messages_flavour(record: dict[str, Any]) -> str:
    openai = anthropic = False
    if "system" in record:
        anthropic = True
    tools = record.get("tools")
    for tool in tools if isinstance(tools, list) else []:
        if isinstance(tool, dict):
            if tool.get("type") == "function" or "function" in tool:
                openai = True
            if "input_schema" in tool:
                anthropic = True
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if (isinstance(role, str) and role in _OPENAI_ROLES) or _OPENAI_KEYS & set(msg):
            openai = True
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                kind = block.get("type") if isinstance(block, dict) else None
                if not isinstance(kind, str):
                    continue
                if kind in _OPENAI_PARTS:
                    openai = True
                elif kind in _ANTHROPIC_BLOCKS:
                    anthropic = True
    if anthropic and not openai:
        return "anthropic"
    return "openai"


def detect(record: Any) -> str | None:
    """Return the name of the format ``record`` looks like, or ``None``.

    Records with a ``messages`` list are OpenAI unless they show a sign of the Anthropic
    Messages API (a top-level ``system``, ``tool_use``/``tool_result``/``image`` blocks, or
    ``input_schema`` tools) and no OpenAI sign. Plain user/assistant string messages read the
    same in both, and are reported as ``openai``.
    """
    if not isinstance(record, dict):
        return None
    convs = record.get("conversations")
    if isinstance(convs, list) and all(
        isinstance(e, dict) and "from" in e and "value" in e for e in convs
    ):
        return "sharegpt"
    if isinstance(record.get("contents"), list):
        return "gemini"
    if isinstance(record.get("messages"), list):
        return _messages_flavour(record)
    if "instruction" in record and "output" in record:
        return "alpaca"
    text = record.get("text")
    if isinstance(text, str) and "<|im_start|>" in text:
        return "chatml"
    return None
