"""HotkeyManager: single persistent listener, runtime rebind, own combo
matching, and key-capture mode. All tests drive _on_press/_on_release
directly — no real pynput Listener is ever started."""
import pytest
from pynput import keyboard

from whisperflow_local.hotkeys import (
    HotkeyManager,
    combo_string,
    key_name,
    parse_combo,
    resolve_key,
)


def make_mgr(ptt="alt_r", combo="<cmd>+<shift>+d"):
    counts = {"down": 0, "up": 0, "toggle": 0}
    mgr = HotkeyManager(
        ptt,
        combo,
        on_ptt_down=lambda: counts.__setitem__("down", counts["down"] + 1),
        on_ptt_up=lambda: counts.__setitem__("up", counts["up"] + 1),
        on_toggle=lambda: counts.__setitem__("toggle", counts["toggle"] + 1),
    )
    return mgr, counts


D = keyboard.KeyCode.from_char("d")


# ------------------------------------------------------------ helpers

def test_resolve_key_special_and_char():
    assert resolve_key("alt_r") == keyboard.Key.alt_r
    assert resolve_key("f18") == keyboard.Key.f18
    assert resolve_key("x") == keyboard.KeyCode.from_char("x")
    with pytest.raises(ValueError):
        resolve_key("not_a_key")


def test_key_name_roundtrip():
    assert key_name(keyboard.Key.alt_r) == "alt_r"
    assert key_name(keyboard.Key.f18) == "f18"
    assert key_name(keyboard.KeyCode.from_char("A")) == "a"


def test_parse_combo():
    mods, trigger = parse_combo("<cmd>+<shift>+d")
    assert mods == frozenset({"cmd", "shift"})
    assert trigger == D
    mods, trigger = parse_combo("<alt>+<f5>")
    assert mods == frozenset({"alt"})
    assert trigger == keyboard.Key.f5
    assert parse_combo("") == (None, None)
    with pytest.raises(ValueError):
        parse_combo("<cmd>+<shift>")  # no trigger key
    with pytest.raises(ValueError):
        parse_combo("<cmd>+nope")


def test_combo_string_canonical_order():
    assert combo_string({"shift", "cmd"}, "d") == "<cmd>+<shift>+d"
    assert combo_string(set(), "f18") == "<f18>"


# ------------------------------------------------------------ push-to-talk

def test_ptt_press_release():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == 1
    mgr._on_press(keyboard.Key.alt_r)  # macOS auto-repeat
    assert counts["down"] == 1
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == 1
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == 1


def test_other_keys_do_not_fire_ptt():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_l)
    mgr._on_press(D)
    assert counts["down"] == 0


def test_rebind_swaps_targets_without_touching_listener():
    mgr, counts = make_mgr()
    sentinel = object()
    mgr._listener = sentinel  # a restart would replace or stop this
    mgr.update("f18", "<cmd>+<shift>+d")
    assert mgr._listener is sentinel
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == 0
    mgr._on_press(keyboard.Key.f18)
    assert counts["down"] == 1
    mgr._on_release(keyboard.Key.f18)
    assert counts["up"] == 1


# ------------------------------------------------------------ toggle combo

def test_combo_fires_with_exact_modifiers():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.cmd_l)
    mgr._on_press(keyboard.Key.shift_l)
    mgr._on_press(D)
    assert counts["toggle"] == 1
    mgr._on_release(D)
    mgr._on_press(D)
    assert counts["toggle"] == 2


def test_combo_wrong_or_extra_modifiers_do_not_fire():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.cmd_l)
    mgr._on_press(D)  # missing shift
    assert counts["toggle"] == 0
    mgr._on_release(D)
    mgr._on_press(keyboard.Key.shift_l)
    mgr._on_press(keyboard.Key.alt_l)
    mgr._on_press(D)  # extra alt
    assert counts["toggle"] == 0


def test_combo_does_not_repeat_while_trigger_held():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.cmd_l)
    mgr._on_press(keyboard.Key.shift_l)
    mgr._on_press(D)
    mgr._on_press(D)  # auto-repeat
    assert counts["toggle"] == 1


def test_combo_modifier_release_resets_state():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.cmd_l)
    mgr._on_release(keyboard.Key.cmd_l)
    mgr._on_press(keyboard.Key.shift_l)
    mgr._on_press(D)
    assert counts["toggle"] == 0


def test_combo_rebind_at_runtime():
    mgr, counts = make_mgr()
    mgr.update("alt_r", "<ctrl>+x")
    mgr._on_press(keyboard.Key.cmd_l)
    mgr._on_press(keyboard.Key.shift_l)
    mgr._on_press(D)
    assert counts["toggle"] == 0
    for k in (D, keyboard.Key.shift_l, keyboard.Key.cmd_l):
        mgr._on_release(k)
    mgr._on_press(keyboard.Key.ctrl_l)
    mgr._on_press(keyboard.KeyCode.from_char("x"))
    assert counts["toggle"] == 1


# ------------------------------------------------------------ capture mode

def test_capture_ptt_takes_next_key_including_bare_modifier():
    mgr, counts = make_mgr()
    session = mgr.begin_capture("ptt")
    assert session.state == "waiting"
    mgr._on_press(keyboard.Key.f19)
    assert session.state == "done"
    assert session.result == "f19"
    # capture is one-shot: normal handling resumes afterwards
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == 1


def test_capture_ptt_bare_modifier():
    mgr, _ = make_mgr()
    session = mgr.begin_capture("ptt")
    mgr._on_press(keyboard.Key.cmd_r)
    assert session.state == "done"
    assert session.result == "cmd_r"


def test_capture_suppresses_normal_callbacks():
    mgr, counts = make_mgr()
    mgr.begin_capture("ptt")
    mgr._on_press(keyboard.Key.alt_r)  # is the current PTT key
    assert counts["down"] == 0


def test_capture_esc_cancels():
    mgr, _ = make_mgr()
    session = mgr.begin_capture("ptt")
    mgr._on_press(keyboard.Key.esc)
    assert session.state == "cancelled"
    assert session.result is None


def test_capture_toggle_combo():
    mgr, counts = make_mgr()
    session = mgr.begin_capture("toggle")
    mgr._on_press(keyboard.Key.cmd_l)
    mgr._on_press(keyboard.Key.shift_l)
    assert session.state == "waiting"  # bare modifiers keep waiting
    mgr._on_press(D)
    assert session.state == "done"
    assert session.result == "<cmd>+<shift>+d"
    assert counts["toggle"] == 0  # suppressed during capture


def test_cancel_capture():
    mgr, _ = make_mgr()
    session = mgr.begin_capture("ptt")
    mgr.cancel_capture()
    assert session.state == "cancelled"


# ------------------------------------------------------------ keycap labels

def test_pretty_key_and_combo():
    from whisperflow_local.keycap import pretty_combo, pretty_key
    assert pretty_key("alt_r") == "Right ⌥ Option"
    assert pretty_key("f18") == "F18"
    assert pretty_key("a") == "A"
    assert pretty_combo("<cmd>+<shift>+d") == "⌘⇧D"
    assert pretty_combo("<ctrl>+<f5>") == "⌃F5"
