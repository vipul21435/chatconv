"""Download the public datasets used for the README numbers and write them as JSONL.

Usage: python scripts/fetch_data.py DEST_DIR

Nothing downloaded here is committed. Each source is listed with the licence stated on its
dataset card.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets"
ROWS = "https://datasets-server.huggingface.co/rows"

FILES = {
    # ShareGPT with function_call / observation entries (Apache-2.0)
    "glaive_toolcall_en.jsonl": ("llamafactory/glaive_toolcall_en", "glaive_toolcall_en_1k.json"),
    # ShareGPT with system / human / gpt / tool entries and <tool_call> tags (Apache-2.0)
    "hermes_func_calling_singleturn.jsonl": (
        "NousResearch/hermes-function-calling-v1",
        "func-calling-singleturn.json",
    ),
    # Alpaca instruction / input / output (CC BY 4.0)
    "alpaca_cleaned.jsonl": ("yahma/alpaca-cleaned", "alpaca_data_cleaned.json"),
}

# OpenAI-style messages with tool_calls, via the dataset viewer API (Apache-2.0)
PAGED = {"toolace_parsed.jsonl": ("minpeter/toolace-parsed", "toolace", "train", 2500)}


def get(url: str, tries: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chatconv-fetch"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == tries - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def main(dest: str) -> None:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name, (repo, filename) in FILES.items():
        rows = json.loads(get(f"{HF}/{repo}/resolve/main/{filename}"))
        with open(out / name, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} records")
    for name, (dataset, config, split, limit) in PAGED.items():
        n = 0
        with open(out / name, "w", encoding="utf-8") as fh:
            for offset in range(0, limit, 100):
                url = (
                    f"{ROWS}?dataset={dataset}&config={config}&split={split}"
                    f"&offset={offset}&length=100"
                )
                page = json.loads(get(url))
                for item in page["rows"]:
                    fh.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
                    n += 1
                time.sleep(1.0)
        print(f"{name}: {n} records")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
