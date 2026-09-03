#!/usr/bin/env python
"""Drive the clarify card standalone to eyeball the open/print/close motion.

    .venv/bin/python scripts/clarify_demo.py

Shows question 1, swaps to question 2 (different option count, so the height
animates), then closes. No LLM, no mic, no hotkeys.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from AppKit import (  # noqa: E402
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSRunLoop,
    NSRunLoopCommonModes,
    NSTimer,
)

from whisperflow_local.clarify import ClarifyPanel  # noqa: E402

app = NSApplication.sharedApplication()
app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
panel = ClarifyPanel()

HINT = "撳 1-3 揀  ·  esc 取消"
OTHER = "或者喺度打你嘅答案…"
steps = [
    (0.3, lambda: panel.show("對象係邊？", ["Dropbox", "開發團隊"], HINT, OTHER)),
    (3.0, lambda: panel.show("用咩語氣？", ["正式", "簡約", "親切"], HINT, OTHER)),
    (6.0, lambda: panel.begin_hide()),
    (7.2, lambda: app.terminate_(None)),
]
for delay, fn in steps:
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        delay, False, lambda t, fn=fn: fn())

for t in NSRunLoop.currentRunLoop().currentMode(), :
    pass
print("clarify demo: 開 → 換問題 → 收，約 7 秒")
app.run()
