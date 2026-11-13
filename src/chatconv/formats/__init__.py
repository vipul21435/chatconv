"""Format registry."""

from __future__ import annotations

from .anthropic import Anthropic
from .base import Format, FormatError
from .gemini import Gemini
from .openai import OpenAI
from .sharegpt import ShareGPT

FORMATS: dict[str, Format] = {f.name: f for f in (OpenAI(), Anthropic(), Gemini(), ShareGPT())}


def get_format(name: str) -> Format:
    try:
        return FORMATS[name]
    except KeyError:
        raise FormatError(f"unknown format {name!r}; known: {', '.join(FORMATS)}") from None


__all__ = ["FORMATS", "Format", "FormatError", "get_format"]
