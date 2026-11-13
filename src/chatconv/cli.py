"""Command-line interface: ``chatconv convert | detect | losses``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from typing import IO

from . import __version__
from .convert import Report, convert_stream, read_jsonl
from .detect import detect
from .formats import FORMATS
from .losses import LOSS_CODES, Policy


def _open_in(stack: ExitStack, path: str) -> IO[str]:
    if path == "-":
        return sys.stdin
    return stack.enter_context(open(path, encoding="utf-8"))


def _open_out(stack: ExitStack, path: str | None) -> IO[str] | None:
    if path is None:
        return None
    if path == "-":
        return sys.stdout
    return stack.enter_context(open(path, "w", encoding="utf-8"))


def _dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _policy(args: argparse.Namespace) -> Policy:
    return Policy(
        system=args.system,
        images=args.images,
        system_turns=args.system_turns,
        extra=args.extra,
    )


def _add_conversion_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="JSONL file, or - for standard input")
    p.add_argument("--to", required=True, choices=sorted(FORMATS), help="target format")
    p.add_argument(
        "--from",
        dest="source",
        default="auto",
        choices=["auto", *sorted(FORMATS)],
        help="source format (default: detect per record)",
    )
    g = p.add_argument_group("policies for what the target cannot hold")
    g.add_argument(
        "--system",
        choices=["fold", "drop"],
        default="fold",
        help="no system slot: fold into the first user turn, or drop",
    )
    g.add_argument(
        "--images",
        choices=["drop", "placeholder"],
        default="drop",
        help="no image parts: drop them, or write a text placeholder",
    )
    g.add_argument(
        "--system-turns",
        choices=["user", "drop"],
        default="user",
        help="no later system messages: write them as user turns, or drop",
    )
    g.add_argument(
        "--extra",
        choices=["keep", "drop"],
        default="keep",
        help="unknown fields: keep where the target has room, or drop all",
    )


def _print_report(report: Report, seconds: float, out: IO[str]) -> None:
    rate = report.records / seconds if seconds > 0 else 0.0
    print(
        f"records {report.records}  converted {report.converted}  "
        f"lossless {report.lossless}  rejected {report.rejected}  ({rate:,.0f} records/s)",
        file=out,
    )
    if report.sources:
        print(
            "sources: " + ", ".join(f"{k} {v}" for k, v in report.sources.most_common()), file=out
        )
    for code, n in report.records_with.most_common():
        share = n / report.records if report.records else 0.0
        print(f"  {code:<26} {n:>8} records ({share:6.1%})  {LOSS_CODES[code]}", file=out)
    for message, n in report.errors.most_common(10):
        print(f"  rejected: {message} ({n})", file=out)


def cmd_convert(args: argparse.Namespace) -> int:
    report = Report()
    start = time.perf_counter()
    with ExitStack() as stack:
        src = _open_in(stack, args.input)
        out = _open_out(stack, args.output) or sys.stdout
        rejects = _open_out(stack, args.rejects)
        loss_log = _open_out(stack, args.losses)
        items = convert_stream(read_jsonl(src), args.to, args.source, _policy(args), args.strict)
        for item in items:
            report.add(item)
            if loss_log is not None and item.losses:
                entry = {"line": item.line, "losses": [loss.to_dict() for loss in item.losses]}
                loss_log.write(_dump(entry) + "\n")
            if item.ok:
                out.write(_dump(item.record) + "\n")
                continue
            print(f"line {item.line}: {item.error}", file=sys.stderr)
            if rejects is not None:
                rejects.write(
                    _dump({"line": item.line, "error": item.error, "record": item.record}) + "\n"
                )
    _print_report(report, time.perf_counter() - start, sys.stderr)
    return 1 if report.rejected else 0


def cmd_losses(args: argparse.Namespace) -> int:
    report = Report()
    start = time.perf_counter()
    with ExitStack() as stack:
        src = _open_in(stack, args.input)
        for item in convert_stream(read_jsonl(src), args.to, args.source, _policy(args)):
            report.add(item)
            if args.per_record and (item.losses or not item.ok):
                codes = sorted({loss.code for loss in item.losses})
                print(f"{item.line}\t{','.join(codes) or 'error: ' + str(item.error)}")
    seconds = time.perf_counter() - start
    if args.json:
        print(_dump({**report.to_dict(), "seconds": round(seconds, 3)}))
    else:
        _print_report(report, seconds, sys.stdout)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    counts: Counter[str] = Counter()
    with ExitStack() as stack:
        for n, record, error in read_jsonl(_open_in(stack, args.input)):
            name = "invalid-json" if error else (detect(record) or "unknown")
            counts[name] += 1
            if args.per_record:
                print(f"{n}\t{name}")
    if not args.per_record:
        for name, n in counts.most_common():
            print(f"{name}\t{n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatconv",
        description="Convert chat datasets between formats and report what is lost.",
    )
    parser.add_argument("--version", action="version", version=f"chatconv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert a JSONL file to another format")
    _add_conversion_args(p)
    p.add_argument("-o", "--output", help="output JSONL (default: standard output)")
    p.add_argument("--rejects", help="write records that could not be converted here")
    p.add_argument("--losses", help="write per-record losses here as JSONL")
    p.add_argument("--strict", action="store_true", help="reject any record that loses anything")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("losses", help="report what a conversion would lose, without writing it")
    _add_conversion_args(p)
    p.add_argument("--per-record", action="store_true", help="one line per lossy record")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.set_defaults(func=cmd_losses)

    p = sub.add_parser("detect", help="detect the format of each record")
    p.add_argument("input", help="JSONL file, or - for standard input")
    p.add_argument("--per-record", action="store_true", help="print line number and format")
    p.set_defaults(func=cmd_detect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
