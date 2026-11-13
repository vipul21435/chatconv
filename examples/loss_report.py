"""Write one tool-calling conversation in every format and print what each one loses.

Run from the repository root:  python examples/loss_report.py
"""

from chatconv.formats import FORMATS
from chatconv.model import Conversation, Image, Text, ToolCall, ToolResult, Turn

conv = Conversation(
    system=[Text("You help with home repairs.")],
    turns=[
        Turn(
            "user",
            [Text("What size is this bolt?"), Image("https://example.org/bolt.jpg", "high")],
            name="sam",
        ),
        Turn(
            "assistant",
            [Text("Let me measure it."), ToolCall("measure_bolt", {"view": "side"}, "m1")],
        ),
        Turn("tool", [ToolResult([Text("M8 x 40")], "m1", "measure_bolt")]),
        Turn("assistant", [Text("It is an M8 bolt, 40 mm long.")]),
    ],
)

for name, fmt in FORMATS.items():
    _, losses = fmt.write(conv)
    codes = sorted({loss.code for loss in losses})
    print(f"{name:<10} {len(losses):>2} loss(es)  {', '.join(codes) or '-'}")
