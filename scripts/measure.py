"""Convert each downloaded dataset to every format and print a Markdown table of the results.

Usage: python scripts/measure.py DATA_DIR

For each dataset and target: records read, share converted without any loss, records
rejected (could not be read), the three most common loss reasons with the share of records
they touch, and records per second for read plus write (best of three passes, one process).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

from chatconv.convert import convert_stream, read_jsonl
from chatconv.formats import FORMATS

DATASETS = [
    ("glaive_toolcall_en.jsonl", "llamafactory/glaive_toolcall_en", "ShareGPT, function_call"),
    (
        "hermes_func_calling_singleturn.jsonl",
        "NousResearch/hermes-function-calling-v1",
        "ShareGPT, tagged calls",
    ),
    ("toolace_parsed.jsonl", "minpeter/toolace-parsed", "OpenAI messages"),
    ("alpaca_cleaned.jsonl", "yahma/alpaca-cleaned", "Alpaca"),
]


def run(lines: list[str], target: str) -> tuple[Counter, Counter, int, float]:
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        with_loss: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        sources = Counter()
        for item in convert_stream(read_jsonl(lines), target):
            if not item.ok:
                totals["rejected"] += 1
                continue
            sources[item.source] += 1
            totals["converted"] += 1
            if not item.losses:
                totals["lossless"] += 1
            with_loss.update({loss.code for loss in item.losses})
        best = min(best, time.perf_counter() - start)
    return totals, with_loss, sum(sources.values()), len(lines) / best


def main(data_dir: str) -> None:
    root = Path(data_dir)
    print("| dataset | target | records | lossless | rejected | most common losses | rec/s |")
    print("|---|---|---:|---:|---:|---|---:|")
    for filename, name, kind in DATASETS:
        lines = (root / filename).read_text(encoding="utf-8").splitlines(keepends=True)
        for target in FORMATS:
            totals, with_loss, _, rate = run(lines, target)
            n = len(lines)
            top = ", ".join(f"{code} {k / n:.0%}" for code, k in with_loss.most_common(3))
            print(
                f"| {name} ({kind}) | {target} | {n} | {totals['lossless'] / n:.1%} | "
                f"{totals['rejected']} | {top or '-'} | {rate:,.0f} |"
            )
    rejects = Counter()
    for filename, name, _ in DATASETS:
        lines = (root / filename).read_text(encoding="utf-8").splitlines(keepends=True)
        for item in convert_stream(read_jsonl(lines), "openai"):
            if not item.ok:
                rejects[(name, item.error)] += 1
    if rejects:
        print()
        print("Rejected records (reading, target openai):")
        for (name, error), k in rejects.most_common(10):
            print(f"- {name}: {k} x {json.dumps(error)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
