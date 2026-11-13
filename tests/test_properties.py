"""Write then read is the identity on what the target holds, and the loss list is exact."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from chatconv.formats import FORMATS
from chatconv.losses import Policy, run_steps

from .strategies import conversations

policies = st.builds(
    Policy,
    system=st.sampled_from(["fold", "drop"]),
    images=st.sampled_from(["drop", "placeholder"]),
    system_turns=st.sampled_from(["user", "drop"]),
    extra=st.sampled_from(["keep", "drop"]),
)


@pytest.mark.parametrize("name", sorted(FORMATS))
@given(conv=conversations(), policy=policies)
def test_losses_are_exact(name, conv, policy):
    fmt = FORMATS[name]
    record, losses = fmt.write(conv, policy)
    record = json.loads(json.dumps(record))  # what a JSONL file would hold
    back = fmt.read(record)
    assert back == fmt.fit(conv, losses, policy)
    if not losses:
        assert back == conv


@pytest.mark.parametrize("name", sorted(FORMATS))
@given(conv=conversations(), policy=policies)
def test_every_step_reports_each_change(name, conv, policy):
    """A step that reports nothing must leave the conversation unchanged."""
    fmt = FORMATS[name]
    current = conv.copy()
    for step in fmt.steps:
        after, losses = run_steps(current, [step], policy)
        if not losses:
            assert after == current, step.name
        current = after


@pytest.mark.parametrize("name", sorted(FORMATS))
@given(conv=conversations(), policy=policies)
def test_fitted_conversation_is_a_fixed_point(name, conv, policy):
    fmt = FORMATS[name]
    record, losses = fmt.write(conv, policy)
    back = fmt.read(record)
    again, more = fmt.write(back, policy)
    assert more == []
    assert again == record
