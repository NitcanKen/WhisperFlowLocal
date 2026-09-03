"""HotkeyManager: single persistent listener, runtime rebind, own combo
matching, and key-capture mode. All tests drive _on_press/_on_release
directly — no real pynput Listener is ever started."""
import threading
import time

import pytest
from pynput import keyboard

from whisperflow_local.hotkeys import (
    HotkeyManager,
    combo_string,
    context_mods,
    hold_matches,
    key_name,
    parse_combo,
    parse_hold_combo,
    resolve_key,
    retarget_hold,
)


def make_mgr(ptt="alt_r", combo="<cmd>+<shift>+d", gen="<shift>+<alt_r>"):
    # down/up collect the MODE of each hold session, so the tests assert what
    # was started, not just how many times.
    counts = {"down": [], "up": [], "toggle": 0}
    mgr = HotkeyManager(
        ptt,
        combo,
        gen,
        on_ptt_down=lambda mode: counts["down"].append(mode),
        on_ptt_up=lambda mode: counts["up"].append(mode),
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
    assert counts["down"] == ["dictate"]
    mgr._on_press(keyboard.Key.alt_r)  # macOS auto-repeat
    assert counts["down"] == ["dictate"]
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == ["dictate"]
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == ["dictate"]


def test_other_keys_do_not_fire_ptt():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_l)
    mgr._on_press(D)
    assert counts["down"] == []


def test_rebind_swaps_targets_without_touching_listener():
    mgr, counts = make_mgr()
    sentinel = object()
    mgr._listener = sentinel  # a restart would replace or stop this
    mgr.update("f18", "<cmd>+<shift>+d")
    assert mgr._listener is sentinel
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == []
    # alt_r is itself a modifier: release it, or it stays in _mods_held and
    # the (strict) empty-context requirement blocks the next hold.
    mgr._on_release(keyboard.Key.alt_r)
    mgr._on_press(keyboard.Key.f18)
    assert counts["down"] == ["dictate"]
    mgr._on_release(keyboard.Key.f18)
    assert counts["up"] == ["dictate"]


# ------------------------------------------------------------ off-thread dispatch
#
# Root-cause regression: pynput invokes _on_press/_on_release synchronously on
# the macOS CGEventTap thread. macOS silently disables a tap whose callback runs
# longer than ~1 s, and pynput never re-enables it — so a slow recorder.start()/
# stop() (device open, Bluetooth) killed the listener: the key release was never
# delivered, recording never stopped, and re-pressing did nothing until restart.
# Once the pump is running, callbacks MUST run off the tap thread.

def test_slow_callback_does_not_block_the_listener_thread():
    started = threading.Event()
    release = threading.Event()
    done = threading.Event()

    def slow_down(mode):
        started.set()
        release.wait(2.0)
        done.set()

    mgr = HotkeyManager(
        "alt_r", "",
        on_ptt_down=slow_down, on_ptt_up=lambda mode: None, on_toggle=lambda: None,
    )
    mgr._start_pump()
    try:
        t0 = time.perf_counter()
        mgr._on_press(keyboard.Key.alt_r)  # must return at once, not block
        elapsed = time.perf_counter() - t0
        assert started.wait(1.0)     # callback did start (on the pump thread)
        assert elapsed < 0.1         # but _on_press did NOT wait for it
        assert not done.is_set()     # it is still running off the tap thread
    finally:
        release.set()
        assert done.wait(1.0)
        mgr._stop_pump()


def test_dispatched_callbacks_keep_press_then_release_order():
    order = []
    mgr = HotkeyManager(
        "alt_r", "",
        on_ptt_down=lambda mode: order.append("down"),
        on_ptt_up=lambda mode: order.append("up"),
        on_toggle=lambda: None,
    )
    mgr._start_pump()
    try:
        mgr._on_press(keyboard.Key.alt_r)
        mgr._on_release(keyboard.Key.alt_r)
        deadline = time.time() + 1.0
        while len(order) < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert order == ["down", "up"]
    finally:
        mgr._stop_pump()


def test_a_raising_callback_does_not_kill_the_pump():
    order = []
    mgr = HotkeyManager(
        "alt_r", "",
        on_ptt_down=lambda mode: (_ for _ in ()).throw(RuntimeError("boom")),
        on_ptt_up=lambda mode: order.append("up"),
        on_toggle=lambda: None,
    )
    mgr._start_pump()
    try:
        mgr._on_press(keyboard.Key.alt_r)   # raises inside the pump
        mgr._on_release(keyboard.Key.alt_r)  # pump must survive and run this
        deadline = time.time() + 1.0
        while not order and time.time() < deadline:
            time.sleep(0.005)
        assert order == ["up"]
    finally:
        mgr._stop_pump()


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
    assert counts["down"] == ["dictate"]


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
    assert counts["down"] == []


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


# ------------------------------------------------ hold bindings (dictate/generate)
# Both push-to-talk and the generation combo are HELD bindings on the same
# key. _on_press records the pressed key's own modifier in _mods_held BEFORE
# matching, so a bare Right-Option press arrives as {"alt"} — matching must
# therefore compare the modifier CONTEXT, not the raw set.

def test_context_mods_excludes_the_pressed_keys_own_modifier():
    assert context_mods({"alt"}, keyboard.Key.alt_r) == frozenset()
    assert context_mods({"alt", "shift"}, keyboard.Key.alt_r) == frozenset({"shift"})
    assert context_mods({"cmd"}, D) == frozenset({"cmd"})


def test_hold_matches_is_subset_based_not_equality_based():
    # Requiring equality made any stray modifier silently kill the hotkey.
    # The empty (dictate) spec therefore matches anything; disambiguation is
    # the most-specific-first ordering in _build_holds, not the predicate.
    bare, modified = (frozenset(), keyboard.Key.alt_r), ({"shift"}, keyboard.Key.alt_r)
    assert hold_matches({"alt"}, keyboard.Key.alt_r, *bare)
    assert not hold_matches({"alt"}, keyboard.Key.alt_r, *modified)
    assert hold_matches({"alt", "shift"}, keyboard.Key.alt_r, *modified)
    assert hold_matches({"alt", "shift"}, keyboard.Key.alt_r, *bare)
    # a stray extra modifier must not stop the generation binding matching
    assert hold_matches({"alt", "shift", "cmd"}, keyboard.Key.alt_r, *modified)


def test_generate_wins_over_dictate_when_both_match():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.shift)
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == ["generate"]


def test_hold_matches_ignores_a_different_key():
    assert not hold_matches({"alt"}, keyboard.Key.alt_r, frozenset(), keyboard.Key.f18)
    assert not hold_matches({"alt"}, keyboard.Key.alt_r, frozenset(), None)


def test_parse_hold_combo_accepts_a_modifier_trigger():
    assert parse_hold_combo("<shift>+<alt_r>") == (frozenset({"shift"}), keyboard.Key.alt_r)
    assert parse_hold_combo("") == (None, None)


def test_parse_hold_combo_rejects_a_self_colliding_modifier():
    with pytest.raises(ValueError):
        parse_hold_combo("<alt>+<alt_r>")


def test_retarget_hold_follows_a_ptt_rebind():
    assert retarget_hold("<shift>+<alt_r>", "alt_r", "f18") == "<shift>+<f18>"
    # an independent binding is never hijacked
    assert retarget_hold("<ctrl>+<f19>", "alt_r", "f18") == "<ctrl>+<f19>"
    assert retarget_hold("", "alt_r", "f18") == ""


def test_bare_ptt_key_starts_dictation():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate"] and counts["up"] == ["dictate"]


def test_shift_plus_ptt_key_starts_generation_not_dictation():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.shift)
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == ["generate"]
    mgr._on_release(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.shift)
    assert counts["up"] == ["generate"]


def test_releasing_shift_mid_hold_keeps_the_generation_session():
    mgr, counts = make_mgr()
    for k in (keyboard.Key.shift, keyboard.Key.alt_r):
        mgr._on_press(k)
    mgr._on_release(keyboard.Key.shift)   # let go of Shift, keep holding alt_r
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["generate"] and counts["up"] == ["generate"]


def test_shift_pressed_after_the_ptt_key_does_not_switch_mode():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_press(keyboard.Key.shift)
    mgr._on_release(keyboard.Key.shift)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate"] and counts["up"] == ["dictate"]


def test_an_unrelated_modifier_still_dictates():
    # Cmd+Right-Option dictated before this feature existed and still does:
    # a stray held modifier must never make push-to-talk look broken.
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.cmd)
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate"] and counts["up"] == ["dictate"]


def test_generation_hold_suppresses_auto_repeat():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.shift)
    for _ in range(3):
        mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["generate"] and counts["up"] == ["generate"]


def test_only_one_hold_session_at_a_time():
    mgr, counts = make_mgr(gen="<shift>+<f18>")
    mgr._on_press(keyboard.Key.alt_r)             # dictation starts
    mgr._on_press(keyboard.Key.shift)
    mgr._on_press(keyboard.Key.f18)               # ignored: session in flight
    mgr._on_release(keyboard.Key.f18)             # ...so its release fires nothing
    assert counts["down"] == ["dictate"] and counts["up"] == []
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == ["dictate"]


def test_empty_generate_combo_falls_back_to_dictation():
    mgr, counts = make_mgr(gen="")
    mgr._on_press(keyboard.Key.shift)
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.shift)
    assert counts["down"] == ["dictate"]      # generation off, not dead keys
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate", "dictate"]


def test_generate_rebind_at_runtime_keeps_the_listener():
    mgr, counts = make_mgr()
    sentinel = object()
    mgr._listener = sentinel
    mgr.update("alt_r", "", "<ctrl>+<f18>")
    assert mgr._listener is sentinel
    mgr._on_press(keyboard.Key.ctrl)
    mgr._on_press(keyboard.Key.f18)
    assert counts["down"] == ["generate"]


def test_update_clears_a_stuck_hold():
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate"]
    mgr.update("alt_r", "<cmd>+<shift>+d", "<shift>+<alt_r>")
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == []                     # the stuck session was dropped
    mgr._on_press(keyboard.Key.alt_r)             # and a fresh one still works
    assert counts["down"] == ["dictate", "dictate"]


# ---------------------------------------------------- clarify-panel suppression
# While the clarify panel is up it is the key window, so the digits already go
# to it. The listener's only job is to stop a hold or the toggle from firing
# underneath the panel.

def test_suppression_silences_holds_and_toggle():
    mgr, counts = make_mgr()
    mgr.set_suppressed(True)
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    mgr._on_press(keyboard.Key.cmd)
    mgr._on_press(keyboard.Key.shift)
    mgr._on_press(D)
    assert counts["down"] == [] and counts["up"] == [] and counts["toggle"] == 0


def test_suppression_drops_an_in_flight_hold():
    # Otherwise the release after the panel closes would fire an unmatched up.
    mgr, counts = make_mgr()
    mgr._on_press(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate"]
    mgr.set_suppressed(True)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["up"] == []


def test_resuming_after_suppression_restores_dispatch():
    mgr, counts = make_mgr()
    mgr.set_suppressed(True)
    mgr.set_suppressed(False)
    mgr._on_press(keyboard.Key.alt_r)
    mgr._on_release(keyboard.Key.alt_r)
    assert counts["down"] == ["dictate"] and counts["up"] == ["dictate"]


def test_capture_still_wins_over_suppression():
    # A settings re-bind must always be reachable and escapable.
    mgr, counts = make_mgr()
    session = mgr.begin_capture("ptt")
    mgr.set_suppressed(True)
    mgr._on_press(keyboard.Key.f18)
    assert session.state == "done" and session.result == "f18"
    assert counts["down"] == []
