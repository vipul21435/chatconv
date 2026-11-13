"""Loss accounting: what a target format could not carry, and the policies that decide how.

Every writer first runs a fixed list of *steps* over a copy of the canonical conversation. Each
step rewrites the parts the target cannot represent (folding, dropping, merging, inventing ids)
and returns one :class:`Loss` per change. A step that returns no loss must not change anything,
so the losses are an exact account: replaying only the reported steps on the original
conversation gives back what the target will hold (see :func:`apply_losses`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .model import Conversation

#: Every loss code with a one-line description. Writers only emit codes listed here.
LOSS_CODES: dict[str, str] = {
    "system_folded": "system prompt prepended to the first user turn",
    "system_dropped": "system prompt removed",
    "system_turns_dropped": "system messages after the start of the conversation removed",
    "system_turns_as_user": "system messages after the start written as user turns",
    "images_dropped": "image parts removed",
    "images_as_placeholder": "image parts replaced by a text placeholder",
    "image_detail_dropped": "image detail setting removed",
    "names_dropped": "author names on turns removed",
    "text_joined": "several text parts of one turn joined into one string",
    "turn_split": "assistant text and tool calls written as separate turns",
    "turns_merged": "consecutive turns with the same role merged",
    "tool_ids_invented": "tool-call ids created where the source had none",
    "tool_ids_dropped": "tool-call ids removed",
    "tool_result_names_dropped": "tool names on tool results removed",
    "tool_error_flag_dropped": "the error flag on tool results removed",
    "tool_calls_dropped": "tool calls removed",
    "tool_results_dropped": "tool results removed",
    "tools_dropped": "tool declarations removed",
    "extra_dropped": "fields the target format has no place for removed",
}


@dataclass(frozen=True)
class Loss:
    """One thing a conversion could not carry.

    ``turn`` is the index of the affected turn in the conversation as it stood when the step
    ran, or ``None`` for conversation-level losses (system prompt, tool declarations).
    """

    code: str
    turn: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.code not in LOSS_CODES:
            raise ValueError(f"unknown loss code {self.code!r}")

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"code": self.code}
        if self.turn is not None:
            out["turn"] = self.turn
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class Policy:
    """One choice per lossy situation. The defaults keep as much as the target allows."""

    system: str = "fold"
    """Target without a system slot: ``fold`` into the first user turn or ``drop``."""

    images: str = "drop"
    """Target without images: ``drop`` them or write a ``placeholder`` text part."""

    system_turns: str = "user"
    """System messages after the start, target without them: write as ``user`` or ``drop``."""

    extra: str = "keep"
    """Unknown fields read from the source: ``keep`` them in JSON targets or ``drop`` them."""

    image_placeholder: str = "<image>"

    def __post_init__(self) -> None:
        allowed = {
            "system": ("fold", "drop"),
            "images": ("drop", "placeholder"),
            "system_turns": ("user", "drop"),
            "extra": ("keep", "drop"),
        }
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError(f"policy {name} must be one of {values}")


StepFn = Callable[[Conversation, Policy], list[Loss]]


@dataclass(frozen=True)
class Step:
    """A rewrite a writer applies before encoding. It mutates the conversation in place."""

    name: str
    codes: frozenset[str]
    fn: StepFn = field(compare=False)

    def __call__(self, conv: Conversation, policy: Policy) -> list[Loss]:
        losses = self.fn(conv, policy)
        for loss in losses:
            if loss.code not in self.codes:
                raise AssertionError(f"step {self.name} emitted undeclared code {loss.code}")
        return losses


def step(name: str, *codes: str) -> Callable[[StepFn], Step]:
    """Decorator turning a function into a :class:`Step` that may emit ``codes``."""
    for code in codes:
        if code not in LOSS_CODES:
            raise ValueError(f"unknown loss code {code!r}")

    def wrap(fn: StepFn) -> Step:
        return Step(name, frozenset(codes), fn)

    return wrap


def run_steps(
    conv: Conversation, steps: Sequence[Step], policy: Policy
) -> tuple[Conversation, list[Loss]]:
    """Run ``steps`` on a copy of ``conv``; return the rewritten copy and every loss."""
    out = conv.copy()
    losses: list[Loss] = []
    for s in steps:
        losses.extend(s(out, policy))
    return out, losses


def apply_losses(
    conv: Conversation, steps: Sequence[Step], losses: Iterable[Loss], policy: Policy
) -> Conversation:
    """Replay only the steps whose codes appear in ``losses``.

    For an exact loss account this equals the conversation the writer encoded.
    """
    codes = {loss.code for loss in losses}
    chosen = [s for s in steps if s.codes & codes]
    out, _ = run_steps(conv, chosen, policy)
    return out


def summarize(losses: Iterable[Loss]) -> Counter[str]:
    """Count losses by code."""
    return Counter(loss.code for loss in losses)
