"""The canonical conversation model that every reader produces and every writer consumes.

A conversation has system parts, a list of turns and the tool declarations offered to the model.
A turn has a role and typed parts. Fields a reader does not understand go into ``meta`` bags on
the conversation, the turn or the part, so that a writer for the same format can put them back.

Invariants (checked by :func:`validate`):

* roles are ``system``, ``user``, ``assistant`` or ``tool``; a ``system`` turn is a system
  message that appears after the conversation has started;
* ``ToolCall`` parts appear only in assistant turns;
* ``ToolResult`` parts appear only in tool turns, and tool turns hold nothing else;
* ``Image`` parts appear only in user turns and inside tool results;
* text parts are never empty, and no two tool turns are adjacent.

Readers call :func:`canonicalize` to bring what they parsed into this normal form.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

ROLES = ("system", "user", "assistant", "tool")


@dataclass
class Text:
    """A run of text."""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Image:
    """A reference to an image: an ``http(s)`` URL or a ``data:`` URI."""

    url: str
    detail: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A call the assistant makes to a declared tool.

    ``arguments`` is a dict when the source gave a JSON object, and the raw string otherwise
    (unparseable JSON, or JSON that is not an object).
    """

    name: str
    arguments: dict[str, Any] | str
    id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The output of a tool call, linked to the call by ``id`` where the format has ids.

    ``content`` is a list of text and image parts; a plain string result is one text part.
    """

    content: list[Text | Image] = field(default_factory=list)
    id: str | None = None
    name: str | None = None
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


Part = Text | Image | ToolCall | ToolResult


@dataclass
class ToolSpec:
    """A tool declaration: a name, a description and a JSON schema for the arguments."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """One message: a role, typed parts and an optional author name."""

    role: str
    parts: list[Part] = field(default_factory=list)
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self, sep: str = "\n") -> str:
        """The turn's text parts joined with ``sep``."""
        return sep.join(p.text for p in self.parts if isinstance(p, Text))


@dataclass
class Conversation:
    """A whole conversation in canonical form."""

    turns: list[Turn] = field(default_factory=list)
    system: list[Text] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> Conversation:
        return copy.deepcopy(self)


class InvalidConversation(ValueError):
    """Raised when a conversation breaks one of the canonical invariants."""


def validate(conv: Conversation) -> None:
    """Raise :class:`InvalidConversation` if ``conv`` breaks a canonical invariant."""
    for part in conv.system:
        if not isinstance(part, Text):
            raise InvalidConversation("system parts must be text")
        if not part.text:
            raise InvalidConversation("empty system text part")
    prev_role = None
    for i, turn in enumerate(conv.turns):
        if turn.role not in ROLES:
            raise InvalidConversation(f"turn {i}: unknown role {turn.role!r}")
        if turn.role == "tool" and prev_role == "tool":
            raise InvalidConversation(f"turn {i}: adjacent tool turns")
        if turn.role == "tool" and not turn.parts:
            raise InvalidConversation(f"turn {i}: empty tool turn")
        for part in turn.parts:
            if isinstance(part, Text):
                if not part.text:
                    raise InvalidConversation(f"turn {i}: empty text part")
                if turn.role == "tool":
                    raise InvalidConversation(f"turn {i}: text in a tool turn")
            elif isinstance(part, Image):
                if turn.role != "user":
                    raise InvalidConversation(f"turn {i}: image outside a user turn")
            elif isinstance(part, ToolCall):
                if turn.role != "assistant":
                    raise InvalidConversation(f"turn {i}: tool call outside an assistant turn")
            elif isinstance(part, ToolResult):
                if turn.role != "tool":
                    raise InvalidConversation(f"turn {i}: tool result outside a tool turn")
                for sub in part.content:
                    if not isinstance(sub, (Text, Image)):
                        raise InvalidConversation(f"turn {i}: tool result holds {sub!r}")
                    if isinstance(sub, Text) and not sub.text:
                        raise InvalidConversation(f"turn {i}: empty text in a tool result")
            else:
                raise InvalidConversation(f"turn {i}: unknown part {type(part).__name__}")
        prev_role = turn.role


def _is_plain_system(turn: Turn) -> bool:
    return (
        turn.role == "system"
        and turn.name is None
        and not turn.meta
        and all(isinstance(p, Text) for p in turn.parts)
    )


def parse_arguments(raw: str) -> dict[str, Any] | str:
    """Parse tool-call arguments given as a string: a dict for a JSON object, else ``raw``."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    return value if isinstance(value, dict) else raw


def canonicalize(conv: Conversation) -> Conversation:
    """Bring a freshly parsed conversation into normal form, in place, and return it.

    None of these rewrites loses information:

    * empty text parts are dropped (an empty string carries nothing);
    * plain system turns before the first other turn move into ``conv.system``;
    * adjacent tool turns merge (a run of tool messages is one batch of results);
    * string arguments that hold a JSON object are parsed;
    * a tool result with an id but no name takes the name of the call with that id.
    """
    conv.system = [p for p in conv.system if p.text]
    turns: list[Turn] = []
    leading = True
    for turn in conv.turns:
        turn.parts = [p for p in turn.parts if not (isinstance(p, Text) and not p.text)]
        for part in turn.parts:
            if isinstance(part, ToolResult):
                part.content = [
                    p for p in part.content if not (isinstance(p, Text) and not p.text)
                ]
            elif isinstance(part, ToolCall) and isinstance(part.arguments, str):
                part.arguments = parse_arguments(part.arguments)
        if leading and _is_plain_system(turn):
            conv.system.extend(p for p in turn.parts if isinstance(p, Text))
            continue
        leading = False
        if turn.role == "tool" and turns and turns[-1].role == "tool":
            turns[-1].parts.extend(turn.parts)
            for key, value in turn.meta.items():
                turns[-1].meta.setdefault(key, value)
            continue
        turns.append(turn)
    conv.turns = turns
    names = tool_call_names(conv)
    for turn in conv.turns:
        for part in turn.parts:
            if isinstance(part, ToolResult) and part.name is None and part.id in names:
                part.name = names[part.id]
    return conv


def tool_call_names(conv: Conversation) -> dict[str, str]:
    """Map every tool-call id in ``conv`` to the name of the called tool."""
    names: dict[str, str] = {}
    for turn in conv.turns:
        for part in turn.parts:
            if isinstance(part, ToolCall) and part.id is not None:
                names.setdefault(part.id, part.name)
    return names
