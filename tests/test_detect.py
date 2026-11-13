import pytest

from chatconv.detect import detect


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"conversations": [{"from": "human", "value": "hi"}]}, "sharegpt"),
        ({"messages": [{"role": "user", "content": "hi"}]}, "openai"),
        (
            {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]},
            "openai",
        ),
        ({"system": "be brief", "messages": [{"role": "user", "content": "hi"}]}, "anthropic"),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
                    }
                ]
            },
            "anthropic",
        ),
        (
            {
                "messages": [{"role": "assistant", "content": None, "tool_calls": []}],
                "system": "conflicting signs",
            },
            "openai",
        ),
        ({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}, "gemini"),
        ({"instruction": "Add.", "input": "2 2", "output": "4"}, "alpaca"),
        ({"text": "<|im_start|>user\nhi<|im_end|>\n"}, "chatml"),
        ({"text": "plain prose"}, None),
        ([1, 2], None),
    ],
)
def test_detect(record, expected):
    assert detect(record) == expected
