# chatconv

Convert conversations between common chat-data formats through one typed model, and report
which parts the target format could not carry.

**Status: early.** Two formats work end to end (OpenAI messages and ShareGPT); the others listed
under "Not done yet" are planned and not implemented.

## Why this is fiddly

Chat formats disagree on where the system prompt lives, whether roles must alternate, how a tool
call and its result are written and linked, and whether message content is a string or a list of
typed parts. A converter that maps fields one to one tends to drop tool results, images, author
names or call ids without saying so. chatconv reads every record into one canonical model and,
when it writes a target format, lists each thing it had to fold, merge, invent or drop.

## Install

```
pip install git+https://github.com/vipul21435/chatconv
```

Python 3.10 or newer, no runtime dependencies.

## Quickstart

How much of the OpenAI test fixture survives a conversion to ShareGPT:

```
$ chatconv losses tests/fixtures/openai.jsonl --to sharegpt
records 4  converted 4  lossless 1  rejected 0  (5,790 records/s)
sources: openai 4
  tool_result_names_dropped         2 records ( 50.0%)  tool names on tool results removed
  images_dropped                    1 records ( 25.0%)  image parts removed
  tool_ids_dropped                  1 records ( 25.0%)  tool-call ids removed
  names_dropped                     1 records ( 25.0%)  author names on turns removed
  turn_split                        1 records ( 25.0%)  assistant text and tool calls written as separate turns
```

(The rate on a four-record file says nothing about speed.)

Convert, keeping rejected records and per-record losses in side files:

```
$ chatconv convert tests/fixtures/sharegpt.jsonl --to openai -o out.jsonl \
    --rejects rejects.jsonl --losses losses.jsonl
records 3  converted 3  lossless 1  rejected 0  (3,534 records/s)
sources: sharegpt 3
  tool_ids_invented                 2 records ( 66.7%)  tool-call ids created where the source had none
```

`chatconv detect FILE` counts the detected format of each line (`--per-record` prints them).
Input format is detected per record unless `--from` is given. `--strict` rejects every record
that would lose anything. The exit status is 1 when any record was rejected.

From Python:

```python
from chatconv.formats import get_format

conv = get_format("sharegpt").read(record)
out, losses = get_format("openai").write(conv)
```

`examples/convert_sharegpt.py` runs this on one tool-calling record.

## How it works so far

- **Canonical model** (`chatconv.model`): a conversation holds system text parts, turns and tool
  declarations; a turn has a role (`system`, `user`, `assistant`, `tool`) and typed parts: text,
  image reference, tool call (id, name, arguments) and tool result (id, name, content, error
  flag). Fields a reader does not interpret are kept in `meta` bags on the conversation, turn or
  part, and written back where the target has room.
- **Writers fit, then encode.** Each format has an ordered list of steps that rewrite the
  canonical conversation into what the format can hold, returning one loss per change; encoding
  that fitted conversation loses nothing more.
- **Exact losses.** A step that reports no loss must not change the conversation. So replaying
  only the reported steps on the original gives exactly what reading the output back returns.
  The property tests (hypothesis) check this on generated conversations for every format, and
  check that a conversion with no reported loss reads back unchanged.
- **Tool-call ids.** When the target needs ids and the source has none (ShareGPT to OpenAI),
  ids are invented as `call_<n>` and each result is linked to the first unanswered call of the
  latest assistant turn with the same tool name. OpenAI's older `function_call` form, which
  has no ids, is kept when the conversation already uses it.

Loss codes and their descriptions are in `chatconv.losses.LOSS_CODES`.

## Formats so far

| format | read | write | notes |
|---|---|---|---|
| `openai` | yes | yes | `tool_calls`, `tool` role, legacy `function_call` and `function` role, `image_url` parts; `developer` is read as `system` |
| `sharegpt` | yes | yes | `from`/`value`, `function_call` and `observation` entries, `tools` as a JSON string |

Detection also recognises the shapes of Anthropic messages, Gemini contents, Alpaca and ChatML
records, but there are no readers for them yet, so such records are rejected with that reason.

## Not done yet

- Readers and writers for Anthropic messages, Gemini contents, Alpaca and ChatML text, and the
  one-way Llama 3 and Mistral instruct renderings.
- `chatconv roundtrip`.
- OpenAI `refusal`, audio and file content parts (rejected with an error for now).
- The `--system` and `--system-turns` policies have no effect yet: both current formats hold a
  system prompt and later system messages.
- Results on public datasets.

## Related tools

- [sftlint](https://github.com/vipul21435/sftlint) lints chat and tool-calling datasets;
  chatconv converts them.

## Licence

MIT, see `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
