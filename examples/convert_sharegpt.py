"""Convert one ShareGPT tool-calling record to OpenAI messages and print the losses.

Run from the repository root:  python examples/convert_sharegpt.py
"""

import json

from chatconv.formats import get_format

record = {
    "conversations": [
        {"from": "human", "value": "What is 17 times 23?"},
        {"from": "function_call", "value": '{"name": "multiply", "arguments": {"a": 17, "b": 23}}'},
        {"from": "observation", "value": "391"},
        {"from": "gpt", "value": "17 times 23 is 391."},
    ]
}

conv = get_format("sharegpt").read(record)

for target in ("openai", "sharegpt"):
    out, losses = get_format(target).write(conv)
    print(f"--- {target}: {len(losses)} loss(es)")
    for loss in losses:
        print("  ", loss.to_dict())
    print(json.dumps(out, indent=1)[:600])
