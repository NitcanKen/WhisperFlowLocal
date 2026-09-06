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
- On the existing callback worker, check current macOS session state every
  50 ms (modifier flags for Option/Shift/Control/Command, key bitmap for ordinary
  keys). After observing the current hold as down, recover a missing release
  after 150 ms of consistently released state. Never infer a release from a
  state reader that has not confirmed that hold, or from ambiguous side flags.
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

## Follow-up: immediate stop regression

The user then reported that holding Right Option opened the pill for roughly
one second and immediately closed it. Production logs at 18:31:46 and 18:32:16
showed `hold down` followed immediately by `recovered missed release` and
`hold up`, proving that the new watchdog, not native key-up delivery, ended
these sessions.

The initial implementation queried `CGEventSourceKeyState` for every key.
Native modifiers arrive through `flagsChanged`; the ordinary key bitmap can
remain false while the modifier is held. The live F18 test exercised the
ordinary-key path and did not establish physical Right Option behavior.

The correction reads `CGEventSourceFlagsState` for modifiers and uses their
individual side masks. Aggregate-only modifier flags are unknown, not released.
Each hold must also be positively observed down before watchdog recovery is
armed; this prevents an unavailable/always-false reader from cancelling speech.
Native release events remain effective even when watchdog recovery is unarmed.

Regression coverage combines real Quartz Right Option event decoding with a
flags-state snapshot and an ordinary key bitmap that stays false: a five-second
hold survives, then an omitted release is recovered. It also covers both sides
of all four modifier families, aggregate-only flags, an always-false reader,
and resetting the confirmation on every new hold. These are automated boundary
tests; they do not substitute for a physical-key recording check in the app.

Physical verification after relaunch completed at 18:38. A read-only probe
captured three user-operated Right Option holds: both HID and session modifier
flags stayed at `0x80040` while the ordinary key bitmap remained **false**.
The corrected modifier reader stayed **true** until release, when flags cleared
and it returned false. This directly confirms the API mismatch on this machine.
The app recorded 3.8 s, 1.0 s and 2.7 s respectively; each log showed
`hold state confirmed`, then a native `hold up`, with no recovered-release
message or premature watchdog stop. Full automated suite: **291 passed**.
