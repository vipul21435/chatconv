"""Malformed records are rejected one at a time; they never end the run."""

import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from chatconv.cli import main
from chatconv.convert import convert_stream, roundtrip_stream
from chatconv.detect import detect
from chatconv.formats import FORMATS

FIXTURES = Path(__file__).parent / "fixtures"
SEEDS = [
    json.loads(line)
    for path in sorted(FIXTURES.glob("*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
]
TWO_WAY = [n for n, f in FORMATS.items() if not f.one_way]

json_values = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()),
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=8), inner, max_size=3)
    ),
    max_leaves=8,
)


def paths(value, prefix=()):
    """Every path to a sub-value of a JSON document."""
    yield prefix
    if isinstance(value, dict):
        for key, sub in value.items():
            yield from paths(sub, (*prefix, key))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            yield from paths(sub, (*prefix, i))


def replace(doc, path, new):
    if not path:
        return new
    doc = json.loads(json.dumps(doc))
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = new
    return doc


@st.composite
def mutated(draw):
    seed = draw(st.sampled_from(SEEDS))
    path = draw(st.sampled_from(list(paths(seed))))
    return replace(seed, path, draw(json_values))


def check(record):
    detect(record)
    lines = [(1, record, None)]
    for target in FORMATS:
        for item in convert_stream(lines, target):
            assert item.ok or isinstance(item.error, str)
            if item.ok:
                json.dumps(item.record, ensure_ascii=False).encode("utf-8")
    for target in TWO_WAY:
        for item in roundtrip_stream(lines, target):
            assert item.error is not None or item.exact


@given(record=mutated())
def test_mutated_records_never_crash(record):
    check(record)


@given(record=json_values)
def test_arbitrary_json_never_crashes(record):
    check(record)


def test_probe_lines_are_rejected_with_line_numbers(tmp_path, capsys):
    probes = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "hi"}], "tools": 5},
        {"messages": [{"role": ["x"], "content": "hi"}]},
        {"contents": [{"parts": 5}]},
        {"conversations": [{"from": ["x"], "value": "hi"}]},
        {"system": 3, "messages": []},
    ]
    lines = [json.dumps(p) for p in probes] + [
        '{"messages": [{"role": "user", "content": "\\ud800"}]}'
    ]
    src = tmp_path / "probe.jsonl"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out, rej = tmp_path / "out.jsonl", tmp_path / "rej.jsonl"
    for target in FORMATS:
        code = main(["convert", str(src), "--to", target, "-o", str(out), "--rejects", str(rej)])
        assert code == 1
        rejected = [json.loads(line)["line"] for line in rej.read_text().splitlines()]
        written = out.read_text(encoding="utf-8").splitlines()
        assert len(rejected) + len(written) == len(lines)
        assert 7 in rejected
    capsys.readouterr()
    assert main(["detect", str(src)]) == 0
