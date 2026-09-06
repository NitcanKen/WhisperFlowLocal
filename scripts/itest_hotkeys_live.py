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
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import Quartz  # noqa: E402
from pynput import keyboard  # noqa: E402

from whisperflow_local import applog  # noqa: E402
from whisperflow_local.hotkeys import HotkeyManager  # noqa: E402

# Keep diagnostic events out of the user's production dictation log.
applog.LOG_PATH = os.path.join(tempfile.mkdtemp(prefix="wfl-hotkeys-"), "app.log")
VK_F18 = 79
fired = {"down": 0, "up": 0}

mgr = HotkeyManager(
    "f18", "<cmd>+<shift>+d",
    on_ptt_down=lambda mode: fired.__setitem__("down", fired["down"] + 1),
    on_ptt_up=lambda mode: fired.__setitem__("up", fired["up"] + 1),
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

# Drop one delivery to the manager, while posting the real key-up so the
# macOS session state changes. No further key event is needed for recovery.
tap(True)
time.sleep(0.15)
release_delivered = threading.Event()
original_release = mgr._listener.on_release


def omit_f18_release(key, injected):
    if key == keyboard.Key.f18:
        release_delivered.set()
    else:
        original_release(key, injected)


mgr._listener.on_release = omit_f18_release
tap(False)
assert release_delivered.wait(1), "native release did not reach the listener"
mgr._listener.on_release = original_release
time.sleep(0.4)
assert fired == {"down": 4, "up": 4}, f"missed release not recovered: {fired}"
print("Missed release recovered from real Quartz key state, without another event")

# Disable this process's tap only. A NULL timeout notification must restore
# the SAME tap, and a later physical-event pair must still reach callbacks.
listener = mgr._listener
native_tap = listener._event_tap
Quartz.CGEventTapEnable(native_tap, False)
listener._handler(None, Quartz.kCGEventTapDisabledByTimeout, None, None)
assert Quartz.CGEventTapIsEnabled(native_tap)
assert mgr._listener is listener_obj and listener._event_tap is native_tap
tap(True)
time.sleep(0.3)
assert fired == {"down": 5, "up": 4}, "watchdog ended a key that was still held"
tap(False)
time.sleep(0.3)
assert fired == {"down": 5, "up": 5}, f"tap did not recover: {fired}"
print("Same event tap recovered after timeout; subsequent hold/release succeeded")
mgr.stop()
print("ALIVE: no TSM crash — single-listener design verified end-to-end")
