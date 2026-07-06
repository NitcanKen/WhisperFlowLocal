"""Global hotkeys via ONE persistent pynput listener.

macOS crash this design prevents: creating a pynput keyboard.Listener runs
TSM keyboard-layout calls (TISCopyCurrentKeyboardInputSource) on the new
listener thread; once the app is up, macOS aborts the process with
dispatch_assert_queue_fail (SIGTRAP). So the listener is created once at
startup and NEVER recreated — rebinding hotkeys and key-capture mode only
swap match targets under a lock.

Also owns combo matching (modifier state tracked in-process) — pynput's
GlobalHotKeys would need its own restartable listener, which is exactly
the crash path.

Requires the macOS Input Monitoring permission.
"""
import threading

from pynput import keyboard

# Left/right/plain modifier variants collapse to one canonical name.
MODIFIER_MAP = {}
for _canon, _names in {
    "cmd": ("cmd", "cmd_l", "cmd_r"),
    "ctrl": ("ctrl", "ctrl_l", "ctrl_r"),
    "alt": ("alt", "alt_l", "alt_r", "alt_gr"),
    "shift": ("shift", "shift_l", "shift_r"),
}.items():
    for _n in _names:
        _key = getattr(keyboard.Key, _n, None)
        if _key is not None:
            MODIFIER_MAP[_key] = _canon

MOD_ORDER = ("cmd", "ctrl", "alt", "shift")


def resolve_key(name: str):
    """Turn a config string like 'alt_r', 'cmd_r', 'f18', or 'x' into a
    pynput key object."""
    name = (name or "").strip()
    special = getattr(keyboard.Key, name, None)
    if special is not None:
        return special
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name.lower())
    raise ValueError(f"Unknown key name: {name!r}")


def key_name(key) -> str | None:
    """Inverse of resolve_key: pynput key -> config string (None if the key
    has no stable name, e.g. a dead key)."""
    if isinstance(key, keyboard.Key):
        return key.name
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    return None


def parse_combo(combo: str):
    """'<cmd>+<shift>+d' -> (frozenset({'cmd','shift'}), trigger key object).

    Empty combo -> (None, None) meaning "no toggle hotkey".
    """
    combo = (combo or "").strip()
    if not combo:
        return None, None
    mods, trigger = set(), None
    for token in combo.split("+"):
        token = token.strip()
        if token.startswith("<") and token.endswith(">"):
            inner = token[1:-1]
            if inner in MOD_ORDER:
                mods.add(inner)
                continue
            trigger = resolve_key(inner)  # special trigger, e.g. <f5>
        elif len(token) == 1:
            trigger = keyboard.KeyCode.from_char(token.lower())
        else:
            raise ValueError(f"Bad combo token: {token!r}")
    if trigger is None:
        raise ValueError(f"Combo needs a non-modifier key: {combo!r}")
    return frozenset(mods), trigger


def combo_string(mods, trigger_name: str) -> str:
    """Canonical combo text: modifiers in MOD_ORDER, then the trigger."""
    parts = [f"<{m}>" for m in MOD_ORDER if m in mods]
    parts.append(trigger_name if len(trigger_name) == 1 else f"<{trigger_name}>")
    return "+".join(parts)


def _keys_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    ca, cb = getattr(a, "char", None), getattr(b, "char", None)
    if ca and cb and ca.lower() == cb.lower():
        return True
    va, vb = getattr(a, "vk", None), getattr(b, "vk", None)
    return va is not None and va == vb


class CaptureSession:
    """One-shot key capture result, polled from the main thread.

    state: waiting -> done | cancelled. result is a config string
    (key name for 'ptt', combo text for 'toggle') once done.
    """

    def __init__(self, kind: str):
        self.kind = kind  # "ptt" | "toggle"
        self.state = "waiting"
        self.result = None


class HotkeyManager:
    """Owns the single global listener; dispatches PTT, toggle and capture.

    Callbacks fire on the listener thread — they must not touch AppKit.
    """

    def __init__(self, ptt_key_name: str, toggle_combo: str,
                 on_ptt_down, on_ptt_up, on_toggle):
        self.on_ptt_down = on_ptt_down
        self.on_ptt_up = on_ptt_up
        self.on_toggle = on_toggle
        self._lock = threading.Lock()
        self._ptt_key = resolve_key(ptt_key_name)
        self._combo_mods, self._combo_trigger = parse_combo(toggle_combo)
        self._ptt_held = False
        self._combo_latched = False
        self._mods_held = set()
        self._capture = None
        self._listener = None

    # -- event handling (listener thread) --------------------------------
    def _on_press(self, key):
        mod = MODIFIER_MAP.get(key)
        with self._lock:
            if mod:
                self._mods_held.add(mod)
            capture = self._capture
            if capture is not None and capture.state == "waiting":
                self._handle_capture_press(capture, key, mod)
                return
            fire_down = False
            if _keys_equal(key, self._ptt_key) and not self._ptt_held:
                self._ptt_held = True
                fire_down = True
            fire_toggle = False
            if (self._combo_trigger is not None
                    and _keys_equal(key, self._combo_trigger)
                    and self._mods_held == self._combo_mods
                    and not self._combo_latched):
                self._combo_latched = True
                fire_toggle = True
        if fire_down:
            self.on_ptt_down()
        if fire_toggle:
            self.on_toggle()

    def _on_release(self, key):
        mod = MODIFIER_MAP.get(key)
        with self._lock:
            if mod:
                self._mods_held.discard(mod)
            if self._combo_latched and (
                    _keys_equal(key, self._combo_trigger) or mod):
                self._combo_latched = False
            fire_up = False
            if _keys_equal(key, self._ptt_key) and self._ptt_held:
                self._ptt_held = False
                fire_up = True
        if fire_up:
            self.on_ptt_up()

    def _handle_capture_press(self, capture, key, mod):
        """Called with the lock held; consumes the event."""
        if key == keyboard.Key.esc:
            capture.state = "cancelled"
            self._capture = None
            return
        if capture.kind == "ptt":
            name = key_name(key)
            if name:
                capture.result = name
                capture.state = "done"
                self._capture = None
        else:  # toggle: bare modifiers accumulate, non-modifier finalizes
            if mod:
                return
            name = key_name(key)
            if name:
                capture.result = combo_string(self._mods_held, name)
                capture.state = "done"
                self._capture = None

    # -- capture mode ------------------------------------------------------
    def begin_capture(self, kind: str) -> CaptureSession:
        """Start one-shot capture ('ptt' or 'toggle'); suppresses normal
        dispatch until done/cancelled. Poll session.state from a timer."""
        with self._lock:
            if self._capture is not None:
                self._capture.state = "cancelled"
            self._capture = CaptureSession(kind)
            return self._capture

    def cancel_capture(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.state = "cancelled"
                self._capture = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Create the ONE listener. Called exactly once at app startup."""
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        """App shutdown only — never part of a rebind."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def update(self, ptt_key_name: str, toggle_combo: str) -> None:
        """Re-bind hotkeys at runtime by swapping match targets. The
        listener is intentionally left untouched (see module docstring)."""
        new_ptt = resolve_key(ptt_key_name)
        new_mods, new_trigger = parse_combo(toggle_combo)
        with self._lock:
            self._ptt_key = new_ptt
            self._combo_mods, self._combo_trigger = new_mods, new_trigger
            self._ptt_held = False
            self._combo_latched = False
