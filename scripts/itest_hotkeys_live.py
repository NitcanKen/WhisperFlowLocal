#!/usr/bin/env python
"""Live integration test for the single-listener hotkey design.

Exercises the exact path that used to SIGTRAP the app (listener restart
during rebind): starts a REAL pynput listener, rebinds 6 times at runtime,
posts REAL synthetic key events (Quartz CGEventPost), and asserts the
callbacks fire and the process survives.

Needs Input Monitoring + Accessibility on the hosting terminal.
Uses F18 (vk 79) so the running WhisperFlow app (PTT alt_r) is not tickled.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import Quartz  # noqa: E402

from whisperflow_local.hotkeys import HotkeyManager  # noqa: E402

VK_F18 = 79
fired = {"down": 0, "up": 0}

mgr = HotkeyManager(
    "f18", "<cmd>+<shift>+d",
    on_ptt_down=lambda: fired.__setitem__("down", fired["down"] + 1),
    on_ptt_up=lambda: fired.__setitem__("up", fired["up"] + 1),
    on_toggle=lambda: None,
)
mgr.start()
time.sleep(1.0)
listener_obj = mgr._listener
assert listener_obj is not None, "listener did not start"

# The old design recreated the listener here — that was the SIGTRAP.
for i in range(6):
    mgr.update("alt_r" if i % 2 == 0 else "f18", "<cmd>+<shift>+d")
    time.sleep(0.1)
assert mgr._listener is listener_obj, "listener object was recreated!"
print(f"6 runtime rebinds done; listener object unchanged: "
      f"{mgr._listener is listener_obj}")

# Final binding is f18 — press it for real.
def tap(down: bool) -> None:
    ev = Quartz.CGEventCreateKeyboardEvent(None, VK_F18, down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

for _ in range(3):
    tap(True); time.sleep(0.15); tap(False); time.sleep(0.15)
time.sleep(0.5)

print(f"real key events after rebinds: down={fired['down']} up={fired['up']}")
assert fired["down"] == 3 and fired["up"] == 3, "callbacks did not fire"
mgr.stop()
print("ALIVE: no TSM crash — single-listener design verified end-to-end")
