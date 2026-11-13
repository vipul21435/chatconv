# Changelog

## 0.1.0 (2026-11-13)

First release.

- Canonical conversation model: system parts, turns with typed parts (text, image, tool call,
  tool result) and metadata bags for fields a reader does not interpret.
- Readers and writers for OpenAI messages (including `tool_calls`, the `tool` role and the
  older `function_call` / `function` role), Anthropic messages, Gemini contents, ShareGPT
  (`function_call` / `observation` entries, and `tool` entries with `<tool_call>` tags),
  Alpaca (with `system` and `history`) and ChatML text.
- Write-only renderers for the Llama 3 and Mistral instruct layouts.
- Per-record format detection.
- Loss accounting with 29 loss codes, an exact replay of the reported losses, and five policy
  flags where a target offers a choice (`--system`, `--system-turns`, `--images`, `--tools`,
  `--extra`). Read notes for content a reader keeps without interpreting it.
- Malformed records are rejected one at a time with their line number.
- Commands `convert`, `losses`, `roundtrip` and `detect` over streaming JSONL, with line
  numbers in errors and a side file for rejected records.
