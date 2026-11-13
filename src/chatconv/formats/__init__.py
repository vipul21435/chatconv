"""Format registry."""

from __future__ import annotations

from .base import Format, FormatError
from .openai import OpenAI

FORMATS: dict[str, Format] = {f.name: f for f in (OpenAI(),)}


def get_format(name: str) -> Format:
    try:
        return FORMATS[name]
    except KeyError:
        raise FormatError(f"unknown format {name!r}; known: {', '.join(FORMATS)}") from None


__all__ = ["FORMATS", "Format", "FormatError", "get_format"]
