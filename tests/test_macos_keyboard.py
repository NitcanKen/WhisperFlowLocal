"""Exercise real Quartz events and taps without posting keys to the desktop."""
import pytest
import Quartz
from pynput import keyboard

from whisperflow_local.hotkeys import HotkeyManager
from whisperflow_local.macos_keyboard import MacOSKeyboardListener


def flags_event(vk, flags):
    event = Quartz.CGEventCreateKeyboardEvent(None, vk, True)
    Quartz.CGEventSetType(event, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(event, flags)
    return event


def deliver(listener, vk, flags):
    listener._handle_message(None, Quartz.kCGEventFlagsChanged,
                             flags_event(vk, flags), None, False)


@pytest.mark.parametrize("left,right,left_mask,right_mask,family", [
    (58, 61, 0x20, 0x40, Quartz.kCGEventFlagMaskAlternate),
    (56, 60, 0x02, 0x04, Quartz.kCGEventFlagMaskShift),
    (59, 62, 0x01, 0x2000, Quartz.kCGEventFlagMaskControl),
    (55, 54, 0x08, 0x10, Quartz.kCGEventFlagMaskCommand),
])
@pytest.mark.parametrize("release_right_first", [True, False])
def test_releasing_one_modifier_while_the_other_is_held(
        left, right, left_mask, right_mask, family, release_right_first):
    events = []
    listener = MacOSKeyboardListener(
        on_press=lambda key: events.append(("down", key.value.vk)),
        on_release=lambda key: events.append(("up", key.value.vk)),
    )
    deliver(listener, right, family | right_mask)
    deliver(listener, left, family | left_mask | right_mask)
    first, last = (right, left) if release_right_first else (left, right)
    remaining_mask = left_mask if release_right_first else right_mask
    deliver(listener, first, family | remaining_mask)
    deliver(listener, last, 0)
    assert events == [("down", right), ("down", left), ("up", first), ("up", last)]


def test_right_option_release_reaches_the_active_hold_with_left_option_down():
    events = []
    mgr = HotkeyManager(
        "alt_r", "", on_ptt_down=lambda mode: events.append(("down", mode)),
        on_ptt_up=lambda mode: events.append(("up", mode)), on_toggle=lambda: None,
    )
    listener = MacOSKeyboardListener(on_press=mgr._on_press, on_release=mgr._on_release)
    deliver(listener, 61, 0x80040)
    deliver(listener, 58, 0x80060)
    deliver(listener, 61, 0x80020)
    assert events == [("down", "dictate"), ("up", "dictate")]
    assert mgr._hold_key is None
    assert mgr._mods_held == {"alt"}  # the left side is still held
    deliver(listener, 58, 0)
    assert mgr._mods_held == set()


def test_aggregate_only_synthetic_flags_still_work():
    events = []
    listener = MacOSKeyboardListener(
        on_press=lambda key: events.append("down"),
        on_release=lambda key: events.append("up"),
    )
    deliver(listener, 61, Quartz.kCGEventFlagMaskAlternate)
    deliver(listener, 61, 0)
    assert events == ["down", "up"]


@pytest.mark.parametrize("notification", [Quartz.kCGEventTapDisabledByTimeout,
                                          Quartz.kCGEventTapDisabledByUserInput])
def test_disabled_tap_is_reenabled_without_decoding_a_null_event(notification):
    events, recoveries = [], []
    listener = MacOSKeyboardListener(
        on_press=lambda key: events.append(key),
        on_release=lambda key: events.append(key),
        on_tap_recovered=recoveries.append,
    )
    tap = listener._create_event_tap()
    if tap is None:
        pytest.skip("Input Monitoring permission is required to create a real tap")
    try:
        Quartz.CGEventTapEnable(tap, False)
        assert not Quartz.CGEventTapIsEnabled(tap)
        listener._handler(None, notification, None, None)
        assert Quartz.CGEventTapIsEnabled(tap)
        assert listener._event_tap is tap
        assert len(recoveries) == 1
        assert events == []
    finally:
        Quartz.CFMachPortInvalidate(tap)


def test_unknown_key_state_does_not_claim_a_release():
    assert MacOSKeyboardListener.key_is_pressed(keyboard.KeyCode.from_char("x")) is None
