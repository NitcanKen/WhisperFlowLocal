# Right Option release leaves recording active

## Evidence before the fix

The user reported that releasing Right Option sometimes left the recording
pill open, and further presses did not help until the app was restarted.
The installed backend is pynput 1.8.2. Its Darwin event tap has two verified
failure paths:

1. `ListenerMixin._handler` passes tap-disabled notifications through keyboard
   decoding. It neither retains nor re-enables the disabled tap. Using a real
   Quartz tap, disabling it, and feeding the installed decoder a timeout
   notification left `CGEventTapIsEnabled(tap)` false. Subsequent releases
   cannot reach `HotkeyManager._on_release`, so its hold remains latched.
   Apple documents restoring the existing tap with
   [CGEventTapEnable](https://developer.apple.com/documentation/coregraphics/cgevent/tapenable(tap:enable:)?language=objc).
2. Darwin modifier decoding uses one aggregate Option flag for both sides.
   Feeding real Quartz `flagsChanged` objects through the unmodified decoder
   reproduced: Right Option down → Left Option down → Right Option up → Left
   Option up produces one dictation start and **no stop**. Right Option's
   release is reported as another press while Left Option remains down.

The application stops a hold only on its matching release, with no independent
key-state check. Moving audio callbacks to a FIFO worker had already removed
one source of tap stalls, but did not provide recovery after a lost release
or a disabled tap.

The historical incident cannot be attributed conclusively to one trigger:
the app had already restarted and its old log did not record tap notifications
or hold transitions. The disabled-tap failure fits the reported need to restart;
the two-Option failure alone can recover with another complete Right Option tap.

## Fix and checks

- Keep one listener and the same event tap. Handle disabled notifications before
  touching their possibly-null event, and re-enable the tap. A periodic health
  check also detects disabled taps without a delivered notification.
- Decode modifiers using IOKit's individual left/right device flags; keep
  aggregate-only synthesized events compatible. Track each held modifier side.
- On the existing callback worker, check current macOS session key state every
  50 ms. Recover a missing release after 150 ms of consistently released state.
  This only ends a latched hold and preserves its dictation/generation mode;
  hands-free recording has no hold and is unaffected. Enqueue state transitions
  under the same lock to preserve down/up order during slow audio startup.
- Log hold transitions, tap recovery and recovered releases off the tap thread.

Validation: `python -m pytest tests/ -q` passed 278 tests. The live
`scripts/itest_hotkeys_live.py` also passed: real F18 events after six rebinds,
a deliberately omitted release delivery recovered from real Quartz state
without another event, and a disabled tap restored with subsequent holds
working on the same listener/tap. No ASR/LLM or microphone capture is needed
for these input-path checks.
