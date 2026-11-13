# chatconv

Convert conversations between common chat-data formats through one typed model, and report
exactly which parts the target format could not carry.

## Why this is fiddly

Chat formats disagree on where the system prompt lives (a message, a top-level field, a
separate instruction object, or nowhere), whether roles must alternate, how a tool call and its
result are written and linked (ids, names, or position), and whether message content is a
string or a list of typed parts. A converter that maps fields one to one tends to drop tool
results, images, author names or call ids without saying so, or to invent ids that no longer
line up with their results. chatconv reads every record into one canonical model and, when it
writes the target format, lists each thing it had to fold, merge, invent or drop.

## Install

```
pip install git+https://github.com/vipul21435/chatconv
```

Python 3.10 or newer, no runtime dependencies.

## Quickstart

What converting the Anthropic test fixture to ShareGPT would lose:

```
$ chatconv losses tests/fixtures/anthropic.jsonl --to sharegpt
records 3  converted 3  lossless 0  rejected 0  (2,995 records/s)
sources: anthropic 3
  tool_result_names_dropped         2 records ( 66.7%)  tool names on tool results removed
  tool_ids_dropped                  2 records ( 66.7%)  tool-call ids removed
  turn_split                        1 records ( 33.3%)  assistant text and tool calls written as separate turns
  images_dropped                    1 records ( 33.3%)  image parts removed
  extra_dropped                     1 records ( 33.3%)  fields the target format has no place for removed
  tool_error_flag_dropped           1 records ( 33.3%)  the error flag on tool results removed
```

(Rates on three-record files say nothing about speed; see the results section.)

Round trip: Anthropic to canonical to Gemini to canonical, listing the records that are not
lossless and why:

```
$ chatconv roundtrip tests/fixtures/anthropic.jsonl --to gemini
3	lossy	tool_error_flag_dropped
records 3  lossless 2  lossy 1  rejected 0  mismatch 0
  tool_error_flag_dropped           1 records  the error flag on tool results removed
```

Convert, with rejected records and per-record losses in side files:

```
$ chatconv convert tests/fixtures/sharegpt.jsonl --to anthropic -o out.jsonl \
    --rejects rejects.jsonl --losses losses.jsonl
records 3  converted 3  lossless 1  rejected 0  (2,682 records/s)
sources: sharegpt 3
  tool_ids_invented                 2 records ( 66.7%)  tool-call ids created where the source had none
```

The second output record (tool declarations shortened):

```json
{"messages": [{"role": "user", "content": "What is 17 times 23?"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "call_0", "name": "multiply", "input": {"a": 17, "b": 23}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_0", "content": "391"}]}, {"role": "assistant", "content": "17 times 23 is 391."}], "tools": [...]}
```

The commands:

| command | does |
|---|---|
| `chatconv convert FILE --to F` | write converted JSONL; `--rejects`, `--losses`, `--strict` |
| `chatconv losses FILE --to F` | count what a conversion would lose; `--per-record`, `--json` |
| `chatconv roundtrip FILE --to F` | read, write as F, read back; list lossy records and reasons |
| `chatconv detect FILE` | count the detected format of each line; `--per-record` |

The source format is detected per record unless `--from` names it. Errors carry the input
line number. `convert` exits with 1 when any record was rejected; `--strict` rejects every
record that would lose anything. `roundtrip` exits with 1 on a mismatch (see below).

Policies. Where a target offers a real alternative there is a flag; the other 18 of the 29 loss
codes have no alternative and are only reported:

| flag | choices (default first) | applies when the target |
|---|---|---|
| `--system` | `fold`, `drop` | has no system slot (Mistral): prepend to the first user turn |
| `--system-turns` | `user`, `drop` | has no system message after the start (Gemini, Alpaca, Mistral) |
| `--images` | `drop`, `placeholder` | has no image parts (ShareGPT, Alpaca, text formats; tool results in OpenAI and Gemini) |
| `--tools` | `text`, `drop` | has no tool calls (Alpaca, ChatML, Llama 3, Mistral) |
| `--extra` | `keep`, `drop` | unknown source fields: keep where the target has room, or drop all |

From Python:

```python
from chatconv.formats import get_format

conv = get_format("sharegpt").read(record)
out, losses = get_format("openai").write(conv)
for loss in losses:
    print(loss.code, loss.turn, loss.detail)
```

`examples/convert_sharegpt.py` runs this on a tool-calling record, and
`examples/loss_report.py` writes one conversation in every format and prints the losses.

## How it works

**Canonical model** (`chatconv.model`). A conversation holds system text parts, turns and tool
declarations. A turn has a role (`system`, `user`, `assistant`, `tool`) and typed parts: text,
image reference (URL or `data:` URI, optional detail), tool call (id, name, arguments) and
tool result (id, name, content of text and image parts, error flag). Fields a reader does not
interpret go into `meta` bags on the conversation, turn or part. Readers bring what they parse
into a normal form: empty text dropped, leading plain system messages moved into the system
prompt, adjacent tool turns merged, string arguments holding a JSON object parsed, and a result
with an id but no name given the name of the call with that id.

**Writers fit, then encode.** Each format has an ordered list of steps. A step rewrites the
canonical conversation into something closer to what the format can hold (fold the system
prompt, merge same-role turns, invent ids, write calls as text, drop images) and returns one
loss for every change it makes. Encoding the fitted conversation then loses nothing more.

**Exact losses.** A step that reports no loss must not change the conversation. Then, for a
conversation *c*, target *F* and reported losses *L*,

    read_F(write_F(c)) == replay(c, the steps of F that emitted a code in L)

and *L* empty implies `read_F(write_F(c)) == c`. The property tests check both, with
hypothesis generating canonical conversations (text, images, calls with and without ids,
results, names, metadata, raw-string arguments) under every policy combination, for every
two-way format. They also check that each step reports every change it makes and that a
written record is a fixed point (writing it again loses nothing). `chatconv roundtrip` applies
the same check to real records and reports a `mismatch` if it ever fails.

**Tool-call ids.** Targets that link results by id (OpenAI, Anthropic) get ids invented as
`call_<n>`, skipping ids already in use; a result without an id is linked to the first
unanswered call of the latest assistant turn with the same tool name, or any name when the
result has none. OpenAI's deprecated `function_call` form has no ids; it is written only for
calls that were read from that form (a `function_call` field followed by `function` messages),
never for id-less calls from other sources. Gemini needs a tool name on every result: a result with neither
id nor name takes the name of the call it answers by position (`tool_result_names_inferred`).

**Formats**, from their public documentation:

| format | read | write | notes |
|---|---|---|---|
| `openai` | yes | yes | Chat completions messages and fine-tuning JSONL: `tool_calls`, `tool` role, the deprecated `function_call` and `function` role, `text` and `image_url` parts. `developer` is read as `system`. [1] [2] |
| `anthropic` | yes | yes | Top-level `system`; `text`, `image` (base64 or URL source), `tool_use` and `tool_result` blocks; a user message with tool results becomes a tool turn followed by a user turn. [3] |
| `gemini` | yes | yes | `contents` with `user`/`model` roles; `text`, `inlineData`, `fileData`, `functionCall`, `functionResponse` parts; `systemInstruction`; `tools[].functionDeclarations`. A result's text is written as `{"result": text}`. [4] |
| `sharegpt` | yes | yes | `conversations` of `from`/`value`; `function_call` and `observation` entries; top-level `system` and `tools` [5]. Also `tool` entries and `gpt` values made only of `<tool_call>` blocks, as in [9]. |
| `alpaca` | yes | yes | `instruction`, `input`, `output` [6], plus `system` and `history` [5]. |
| `chatml` | yes | yes | `{"text": "<|im_start|>role\n...<|im_end|>\n..."}` [10] [7]. ChatML names `system`, `user` and `assistant`; the `tool` role (one block per tool result) is chatconv's own convention. |
| `llama3` | no | yes | The chat template of Meta-Llama-3-8B-Instruct [11]: `<|begin_of_text|>`, then per message `<|start_header_id|>role<|end_header_id|>\n\n` + the content with surrounding whitespace stripped + `<|eot_id|>` [8]. One-way. |
| `mistral` | no | yes | The chat template of Mistral-7B-Instruct v0.1 and v0.2 [12]: `<s> [INST] system\n\nuser [/INST] assistant</s> [INST] user [/INST]`. v0.3 and later templates differ and are not rendered. One-way. |

Detection looks at the record's shape: `conversations` of `from`/`value` is ShareGPT,
`contents` is Gemini, `instruction` with `output` is Alpaca, a `text` holding `<|im_start|>` is
ChatML. A `messages` list is Anthropic when it shows an Anthropic sign (top-level `system`,
`tool_use`/`tool_result`/`image` blocks, `input_schema` tools) and no OpenAI sign, and OpenAI
otherwise; plain user/assistant string messages read the same either way.

Loss codes and their descriptions are listed in `chatconv.losses.LOSS_CODES` (29 codes).

## Results on public data

Four public datasets in three source formats, each converted to every format. "lossless" is
the share of records converted with no reported loss, which means the conversation read back
from the output equals, in canonical form, the one read from the source; the output record can
still differ from the input in form (see Limitations). "most common losses" gives the share of
records each loss reason touches. No dataset rows are committed.

```
python scripts/fetch_data.py ~/data/chatconv      # about 70 MB
python scripts/measure.py ~/data/chatconv
```

| source (format, records) | target | lossless | most common losses (share of records) | rec/s |
|---|---|---:|---|---:|
| glaive_toolcall_en, ShareGPT, 1,000 | openai | 48.2% | tool_ids_invented 52% | 4,184 |
|  | anthropic | 48.2% | tool_ids_invented 52% | 4,179 |
|  | gemini | 48.2% | tool_result_names_inferred 52% | 4,169 |
|  | sharegpt | 100.0% | - | 3,733 |
|  | alpaca | 33.0% | tools_dropped 67%, tool_calls_as_text 52%, tool_results_as_text 52% | 3,895 |
|  | chatml | 33.0% | tools_dropped 67%, tool_calls_as_text 52% | 3,820 |
|  | llama3 | 33.0% | tools_dropped 67%, tool_calls_as_text 52%, tool_results_as_text 52% | 4,096 |
|  | mistral | 33.0% | tools_dropped 67%, tool_calls_as_text 52%, tool_results_as_text 52% | 3,831 |
| hermes-function-calling-v1, ShareGPT, 1,893 | openai | 67.7% | tool_ids_invented 32% | 4,094 |
|  | anthropic | 42.4% | tool_ids_invented 58% | 4,872 |
|  | gemini | 100.0% | - | 4,746 |
|  | sharegpt | 100.0% | - | 3,653 |
|  | alpaca | 3.2% | tools_dropped 97%, tool_calls_as_text 58%, text_joined 32% | 4,703 |
|  | chatml | 3.2% | tools_dropped 97%, tool_calls_as_text 58%, text_joined 32% | 3,925 |
|  | llama3 | 3.2% | tools_dropped 97%, tool_calls_as_text 58%, text_joined 32% | 3,872 |
|  | mistral | 0.0% | text_joined 100%, system_folded 100%, tools_dropped 97% | 3,654 |
| toolace-parsed, OpenAI, 2,500 | openai | 73.0% | tool_ids_invented 27% | 2,628 |
|  | anthropic | 17.3% | tool_ids_invented 83%, tool_result_names_dropped 0% | 2,609 |
|  | gemini | 100.0% | - | 2,456 |
|  | sharegpt | 68.2% | tool_result_names_dropped 32% | 2,262 |
|  | alpaca | 10.2% | tools_dropped 90%, tool_calls_as_text 83%, tool_results_as_text 32% | 2,089 |
|  | chatml | 10.2% | tools_dropped 90%, tool_calls_as_text 83%, tool_result_names_dropped 32% | 2,204 |
|  | llama3 | 10.2% | tools_dropped 90%, tool_calls_as_text 83%, tool_results_as_text 32% | 2,112 |
|  | mistral | 10.2% | tools_dropped 90%, tool_calls_as_text 83%, tool_results_as_text 32% | 2,213 |
| alpaca-cleaned, Alpaca, 51,760 | openai | 100.0% | - | 11,300 |
|  | anthropic | 100.0% | - | 8,484 |
|  | gemini | 100.0% | - | 12,024 |
|  | sharegpt | 63.0% | text_joined 37% | 11,592 |
|  | alpaca | 100.0% | - | 11,796 |
|  | chatml | 63.0% | text_joined 37% | 8,287 |
|  | llama3 | 63.0% | text_joined 37% | 13,457 |
|  | mistral | 63.0% | text_joined 37% | 15,670 |

No record of any of the four sets was rejected. `chatconv roundtrip` on every dataset and
two-way target (24 pairs) reported `mismatch 0`, that is, every read-back matched the reported
losses; for example
`chatconv roundtrip ~/data/chatconv/toolace_parsed.jsonl --to anthropic` ends with
`records 2500  lossless 432  lossy 2068  rejected 0  mismatch 0`.

What the numbers say: ShareGPT and OpenAI tool data written for Anthropic needs invented ids
whenever the source had none (the OpenAI rows here have `tool_calls` without ids). Gemini
holds the hermes and ToolACE rows without loss because its call ids are optional, but needs a
tool name on every result, which the glaive observations lack. The text formats
and Alpaca lose every tool declaration; and Alpaca records with a non-empty `input` (37 %)
lose the instruction/input boundary in every format with one string per turn.

Data sources and the licence stated on each dataset card:

- llamafactory/glaive_toolcall_en, 1,000 records, Apache-2.0:
  https://huggingface.co/datasets/llamafactory/glaive_toolcall_en
- NousResearch/hermes-function-calling-v1, `func-calling-singleturn.json`, 1,893 records,
  Apache-2.0: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- minpeter/toolace-parsed, first 2,500 rows, Apache-2.0:
  https://huggingface.co/datasets/minpeter/toolace-parsed
- yahma/alpaca-cleaned, 51,760 records: the card's metadata tag says `cc-by-4.0`, but its
  Licensing Information section says CC BY-NC 4.0, and the upstream Stanford Alpaca README [6]
  says CC BY NC 4.0 (non-commercial use only). Read it as non-commercial. Only counts derived
  from it appear here: https://huggingface.co/datasets/yahma/alpaca-cleaned

## Limitations

- Content types outside the model are rejected with an error rather than carried: OpenAI
  `refusal`, audio and file parts; Anthropic `document`, `thinking`, search-result and server
  tool blocks, and file image sources; Gemini non-image `inlineData` and tools other than
  function declarations.
- "Lossless" compares canonical forms, not bytes. These read-side rewrites count as lossless,
  so even a same-format conversion with no losses can change a record: tool-call arguments
  are re-serialised from parsed JSON (`{"a": 1.0, "b":1e3}` becomes `{"a": 1.0, "b": 1000.0}`);
  string content can become a list of parts or the reverse; keys whose value is `null` are not
  written back; OpenAI's `developer` role comes back as `system`; ShareGPT `user`/`assistant`
  speakers come back as `human`/`gpt`, tagged calls as `function_call` entries and a top-level
  `system` field as a `system` entry; Alpaca always writes `input`.
- Gemini `functionResponse.response` is an object while the canonical result is text: an
  object other than `{"result": <string>}` is read as its JSON text, and results are written
  as `{"result": text}`. Images inside tool results are dropped for OpenAI and Gemini.
- In ShareGPT, a `gpt` value made only of `<tool_call>` blocks of JSON is always read as tool
  calls; `tool` entries keep their text as it is, `<tool_response>` tags included.
- Tool declarations embedded in a system prompt's `<tools>` block are left in the system text;
  only a `tools` field is parsed.
- ChatML has no standard tool-call syntax; calls are written as JSON text per `--tools`. Text
  that contains a format's control tokens is rejected, not escaped.
- The Llama 3 and Mistral renderers are one-way and cover the plain chat layouts only; the
  Llama 3.1 tool roles and newer Mistral tool tokens are not rendered.
- Tool declarations are kept as name, description and JSON schema; schema dialects (for
  example Gemini's `parametersJsonSchema`) are not translated.
- The speeds are from one machine and one run; they are not a comparison with other tools.

## Related tools

- [sftlint](https://github.com/vipul21435/sftlint) lints chat and tool-calling datasets;
  chatconv converts them.

## References

1. OpenAI Python library, chat message types (`chat_completion_message_param.py`,
   `chat_completion_assistant_message_param.py`, `chat_completion_tool_message_param.py`):
   https://github.com/openai/openai-python/tree/main/src/openai/types/chat
2. OpenAI Cookbook, Fine tuning with function-calling (marked archived by OpenAI):
   https://developers.openai.com/cookbook/examples/fine_tuning_for_function_calling
3. Anthropic, Messages API reference: https://platform.claude.com/docs/en/api/messages
4. Google, Gemini API reference (Content, Part, FunctionCall, FunctionResponse):
   https://aistudio.google.com/api-reference/caching
5. LLaMA-Factory, dataset formats: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
6. Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
7. Hugging Face Transformers, chat templates:
   https://huggingface.co/docs/transformers/main/en/chat_templating
8. Meta, Llama 3 model card and prompt formats:
   https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-3/
9. NousResearch/hermes-function-calling-v1, a dataset in ShareGPT form with tagged calls:
   https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
10. OpenAI, Chat Markup Language (ChatML), no longer maintained:
    https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
11. `chat_template` in `tokenizer_config.json` of an ungated copy of Meta-Llama-3-8B-Instruct
    (the original repository needs a login):
    https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
12. `chat_template` in `tokenizer_config.json` of Mistral-7B-Instruct v0.1 and v0.2 (identical):
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json,
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json

## Licence

MIT, see `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
