"""Record-level conversion and streaming over JSONL files."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import IO, Any

from .detect import detect
from .formats import FORMATS, FormatError, get_format
from .losses import Loss, Policy
from .model import InvalidConversation

#: Errors a malformed record can raise inside a reader or writer. They reject that record only.
RECORD_ERRORS = (
    FormatError,
    InvalidConversation,
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
    IndexError,
    RecursionError,
)
_SURROGATE = re.compile(r"\\[uU][dD][89a-fA-F]")


def _error(exc: BaseException) -> str:
    if isinstance(exc, (FormatError, InvalidConversation)):
        return str(exc)
    return f"malformed record ({type(exc).__name__}: {exc})"


@dataclass
class Converted:
    """The outcome of converting one record."""

    line: int
    source: str | None = None
    record: Any = None
    losses: list[Loss] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def convert_record(
    record: Any, target: str, source: str = "auto", policy: Policy | None = None
) -> tuple[Any, list[Loss], str]:
    """Convert one record to ``target``; return the new record, its losses and the source format.

    Raises :class:`FormatError` when the record cannot be read.
    """
    name = detect(record) if source == "auto" else source
    if name is None:
        raise FormatError("format not recognised")
    if name not in FORMATS:
        raise FormatError(f"detected {name}, which has no reader yet")
    conv = get_format(name).read(record)
    out, losses = get_format(target).write(conv, policy)
    return out, losses, name


def read_jsonl(stream: IO[str]) -> Iterator[tuple[int, Any, str | None]]:
    """Yield ``(line_number, record, error)`` for each non-blank line, numbered from 1."""
    for n, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        text = line.rstrip("\r\n")
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            yield n, text, f"invalid JSON: {exc.msg} at column {exc.colno}"
            continue
        except RecursionError:
            yield n, text, "invalid JSON: nested too deeply"
            continue
        if _SURROGATE.search(text):
            try:
                json.dumps(record, ensure_ascii=False).encode("utf-8")
            except UnicodeEncodeError:
                yield n, text, "invalid text: a string holds an unpaired surrogate"
                continue
        yield n, record, None


def convert_stream(
    lines: Iterable[tuple[int, Any, str | None]],
    target: str,
    source: str = "auto",
    policy: Policy | None = None,
    strict: bool = False,
) -> Iterator[Converted]:
    """Convert records one at a time. With ``strict`` a record with any loss is an error."""
    get_format(target)
    for n, record, error in lines:
        if error is not None:
            yield Converted(n, record=record, error=error)
            continue
        try:
            out, losses, name = convert_record(record, target, source, policy)
        except RECORD_ERRORS as exc:
            yield Converted(n, record=record, error=_error(exc))
            continue
        if strict and losses:
            codes = ", ".join(sorted({loss.code for loss in losses}))
            yield Converted(n, name, record, losses, f"strict: lossy ({codes})")
            continue
        yield Converted(n, name, out, losses)


@dataclass
class Report:
    """Totals over a stream of conversions."""

    records: int = 0
    converted: int = 0
    lossless: int = 0
    rejected: int = 0
    sources: Counter[str] = field(default_factory=Counter)
    losses: Counter[str] = field(default_factory=Counter)
    records_with: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)

    def add(self, item: Converted) -> None:
        self.records += 1
        if item.source:
            self.sources[item.source] += 1
        codes = {loss.code for loss in item.losses}
        self.losses.update(loss.code for loss in item.losses)
        self.records_with.update(codes)
        if not item.ok:
            self.rejected += 1
            self.errors[_error_kind(item.error or "")] += 1
            return
        self.converted += 1
        if not item.losses:
            self.lossless += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "converted": self.converted,
            "lossless": self.lossless,
            "rejected": self.rejected,
            "sources": dict(self.sources),
            "losses": dict(self.losses.most_common()),
            "records_with_loss": dict(self.records_with.most_common()),
            "errors": dict(self.errors.most_common()),
        }


def _error_kind(message: str) -> str:
    """Group error messages by their text without positions (``message 3: x`` -> ``x``)."""
    head, sep, tail = message.partition(": ")
    if sep and head.split(" ")[0] in {"message", "entry", "turn"}:
        return tail
    return message.split(" at column ")[0]


@dataclass
class RoundTrip:
    """The outcome of reading a record, writing it as another format and reading it back."""

    line: int
    source: str | None = None
    losses: list[Loss] = field(default_factory=list)
    exact: bool = True
    error: str | None = None

    @property
    def lossless(self) -> bool:
        return self.error is None and self.exact and not self.losses


def roundtrip_stream(
    lines: Iterable[tuple[int, Any, str | None]],
    target: str,
    source: str = "auto",
    policy: Policy | None = None,
) -> Iterator[RoundTrip]:
    """A to canonical to B to canonical, compared with the canonical form after B's losses.

    ``exact`` is false when the read-back conversation differs from what the reported losses
    predict, which would be a bug in chatconv.
    """
    fmt = get_format(target)
    if fmt.one_way:
        raise FormatError(f"{target} is write-only; round trips need a format that reads back")
    for n, record, error in lines:
        if error is not None:
            yield RoundTrip(n, error=error)
            continue
        name = None
        try:
            name = detect(record) if source == "auto" else source
            if name is None or name not in FORMATS:
                raise FormatError(
                    "format not recognised" if name is None else f"no reader for {name}"
                )
            conv = get_format(name).read(record)
            out, losses = fmt.write(conv, policy)
            back = fmt.read(json.loads(json.dumps(out)))
        except RECORD_ERRORS as exc:
            yield RoundTrip(n, name if isinstance(name, str) else None, error=_error(exc))
            continue
        yield RoundTrip(n, name, losses, back == fmt.fit(conv, losses, policy))
