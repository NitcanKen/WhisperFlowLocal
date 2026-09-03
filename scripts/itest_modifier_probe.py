#!/usr/bin/env python
"""Log the RAW pynput events for modifier keys, to diagnose a hold binding.

Read-only: a plain non-suppressing listener in its OWN process, so it never
touches the running app's single-listener invariant and never synthesizes
anything. Press the keys you are debugging; every press/release is written to
the output file with what HotkeyManager would make of it.

    .venv/bin/python scripts/itest_modifier_probe.py [seconds] [outfile]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynput import keyboard  # noqa: E402

from whisperflow_local.hotkeys import (  # noqa: E402
    MODIFIER_MAP,
    context_mods,
    hold_matches,
    parse_hold_combo,
)

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/wfl_probe.log"

gen_mods, gen_key = parse_hold_combo("<shift>+<alt_r>")
ptt_key = keyboard.Key.alt_r
held = set()
out = open(OUT, "w", buffering=1)


def line(s):
    out.write(s + "\n")


def describe(tag, key):
    mod = MODIFIER_MAP.get(key)
    ctx = context_mods(held, key)
    gen = hold_matches(held, key, gen_mods, gen_key)
    dic = hold_matches(held, key, frozenset(), ptt_key)
    line(f"{tag:8} key={key!r:24} vk={getattr(key, 'value', key)!r:28} "
         f"mod={mod!r:8} held={sorted(held)} ctx={sorted(ctx)} "
         f"-> generate={gen} dictate={dic}")


def on_press(key):
    mod = MODIFIER_MAP.get(key)
    if mod:
        held.add(mod)
    describe("PRESS", key)


def on_release(key):
    mod = MODIFIER_MAP.get(key)
    if mod:
        held.discard(mod)
    describe("RELEASE", key)


line(f"probe started, {DURATION:.0f}s — press the keys you are testing")
line(f"generate binding = mods {sorted(gen_mods)} + {gen_key!r}")
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()
time.sleep(DURATION)
listener.stop()
line("probe finished")
out.close()
print(f"wrote {OUT}")
