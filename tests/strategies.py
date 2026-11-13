"""Hypothesis strategies for canonical conversations."""

from hypothesis import strategies as st

from chatconv.model import (
    Conversation,
    Image,
    Text,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    canonicalize,
    validate,
)

words = st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), min_size=1, max_size=12)
names = st.sampled_from(["ana", "ben", "chris"])
tool_names = st.sampled_from(["search", "add", "lookup"])
ids = st.one_of(st.none(), st.sampled_from(["c1", "c2", "c3", "call_0"]))
meta = st.dictionaries(st.sampled_from(["x_a", "x_b"]), st.integers(0, 9), max_size=2)
json_values = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(-100, 100), words),
    lambda inner: st.one_of(st.lists(inner, max_size=3), st.dictionaries(words, inner, max_size=3)),
    max_leaves=6,
)
arguments = st.one_of(
    st.dictionaries(words, json_values, max_size=3), st.sampled_from(["{oops", "[1, 2]", ""])
)

text = st.builds(Text, words, meta)
image = st.builds(
    Image,
    st.sampled_from(["https://example.org/a.png", "data:image/png;base64,iVBORw0KGgo="]),
    st.sampled_from([None, "low", "high"]),
    meta,
)
call = st.builds(ToolCall, tool_names, arguments, ids, meta)
result = st.builds(
    ToolResult,
    st.lists(st.one_of(text, image), max_size=2),
    ids,
    st.one_of(st.none(), tool_names),
    st.booleans(),
    meta,
)


@st.composite
def turn(draw):
    role = draw(st.sampled_from(["user", "assistant", "tool", "system"]))
    name = draw(st.one_of(st.none(), names))
    if role == "user":
        parts = draw(st.lists(st.one_of(text, image), max_size=3))
    elif role == "assistant":
        parts = draw(st.lists(st.one_of(text, call), max_size=3))
    elif role == "system":
        parts = draw(st.lists(text, min_size=1, max_size=2))
    else:
        return Turn("tool", draw(st.lists(result, min_size=1, max_size=2)), None, draw(meta))
    return Turn(role, parts, name, draw(meta))


@st.composite
def conversations(draw):
    spec = st.builds(
        ToolSpec,
        tool_names,
        st.one_of(st.none(), words),
        st.one_of(st.none(), st.just({"type": "object"})),
        meta,
    )
    conv = Conversation(
        turns=draw(st.lists(turn(), max_size=6)),
        system=draw(st.lists(text, max_size=2)),
        tools=draw(st.lists(spec, max_size=2)),
        meta=draw(meta),
    )
    canonicalize(conv)
    validate(conv)
    return conv
