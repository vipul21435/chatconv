"""The interface every format implements."""

from __future__ import annotations

from typing import Any

from ..losses import Loss, Policy, Step, apply_losses, run_steps
from ..model import Conversation, canonicalize, validate


class FormatError(ValueError):
    """A record that the reader cannot turn into a conversation."""


class Format:
    """A chat-data format: a reader, a writer and the steps the writer applies first.

    ``steps`` rewrite a canonical conversation into the subset the format can hold, reporting a
    loss for every change; ``encode`` then writes that subset without further loss, so that
    ``read(encode(c)) == c`` for every ``c`` the steps leave unchanged.
    """

    name: str = ""
    one_way: bool = False
    steps: tuple[Step, ...] = ()

    def decode(self, record: Any) -> Conversation:
        raise NotImplementedError

    def encode(self, conv: Conversation) -> Any:
        raise NotImplementedError

    def read(self, record: Any) -> Conversation:
        """Parse a record into a validated canonical conversation."""
        if self.one_way:
            raise FormatError(f"{self.name} is write-only")
        conv = canonicalize(self.decode(record))
        validate(conv)
        return conv

    def write(self, conv: Conversation, policy: Policy | None = None) -> tuple[Any, list[Loss]]:
        """Write ``conv``; return the record and the losses of the conversion."""
        policy = policy or Policy()
        fitted, losses = run_steps(conv, self.steps, policy)
        return self.encode(fitted), losses

    def fit(self, conv: Conversation, losses: list[Loss], policy: Policy | None = None):
        """The conversation this format holds after the given losses (for checking)."""
        return apply_losses(conv, self.steps, losses, policy or Policy())


def expect(cond: bool, message: str) -> None:
    if not cond:
        raise FormatError(message)


def extras(obj: dict[str, Any], known: set[str]) -> dict[str, Any]:
    """The keys of ``obj`` a reader does not interpret."""
    return {k: v for k, v in obj.items() if k not in known}
