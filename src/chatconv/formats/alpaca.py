"""Alpaca: ``{"instruction", "input", "output"}``, with the optional ``system`` and ``history``
fields (a list of ``[user, assistant]`` pairs) used for multi-turn data.

The last user turn is the instruction; a second text part of that turn is the input. Every
earlier exchange goes into ``history``. There is no place for tools, images, names or turn
metadata, and turns must alternate from user to assistant.
"""

from __future__ import annotations

from typing import Any

from .. import steps as S
from ..losses import Loss, Policy, step
from ..model import Conversation, Text, Turn
from .base import Format, expect, extras

_KEYS = {"instruction", "input", "output", "system", "history"}


@step("alpaca text", "text_joined")
def _fit_text(conv: Conversation, policy: Policy) -> list[Loss]:
    """One string per turn, except the last user turn, which keeps instruction and input."""
    losses: list[Loss] = []
    last_user = len(conv.turns) - 2
    for i, turn in enumerate(conv.turns):
        texts = [p.text for p in turn.parts if isinstance(p, Text)]
        keep = 2 if i == last_user else 1
        if len(texts) > keep:
            head = texts[: keep - 1]
            turn.parts = [Text(t) for t in head] + [Text("\n".join(texts[keep - 1 :]))]
            losses.append(Loss("text_joined", i))
    return losses


class Alpaca(Format):
    name = "alpaca"
    steps = (
        S.drop_meta(
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
            reserved={"conv": _KEYS},
        ),
        S.drop_tools,
        S.tools_to_text,
        S.system_turns,
        S.names,
        S.images(),
        S.join_system,
        S.alternate(first="user", last="assistant"),
        _fit_text,
    )

    def decode(self, record: Any) -> Conversation:
        expect(isinstance(record, dict), "record must be an object")
        for key in ("instruction", "output"):
            expect(isinstance(record.get(key), str), f"{key} must be a string")
        conv = Conversation(meta=extras(record, _KEYS))
        system = record.get("system")
        if system is not None:
            expect(isinstance(system, str), "system must be a string")
            conv.system.append(Text(system))
        history = record.get("history") or []
        expect(isinstance(history, list), "history must be a list")
        for n, pair in enumerate(history):
            expect(
                isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair),
                f"history {n}: must be a [user, assistant] pair of strings",
            )
            conv.turns += [Turn("user", [Text(pair[0])]), Turn("assistant", [Text(pair[1])])]
        user = [Text(record["instruction"])]
        extra_input = record.get("input")
        if extra_input is not None:
            expect(isinstance(extra_input, str), "input must be a string")
            user.append(Text(extra_input))
        conv.turns += [Turn("user", user), Turn("assistant", [Text(record["output"])])]
        return conv

    def encode(self, conv: Conversation) -> dict[str, Any]:
        def text(turn: Turn, k: int = 0) -> str:
            texts = [p.text for p in turn.parts if isinstance(p, Text)]
            return texts[k] if len(texts) > k else ""

        *earlier, user, answer = conv.turns
        record: dict[str, Any] = {
            "instruction": text(user),
            "input": text(user, 1),
            "output": text(answer),
        }
        if conv.system:
            record["system"] = conv.system[0].text
        if earlier:
            record["history"] = [
                [text(u), text(a)] for u, a in zip(earlier[::2], earlier[1::2], strict=True)
            ]
        return {**conv.meta, **record}
