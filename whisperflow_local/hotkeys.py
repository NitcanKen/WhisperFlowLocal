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
import queue
import threading
from functools import partial

from pynput import keyboard

from .applog import log

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


def context_mods(mods_held, key) -> frozenset:
    """Modifiers held *besides* the pressed key's own modifier family.

    _on_press records the pressed key's own modifier in _mods_held BEFORE
    matching runs, so a bare Right-Option press arrives as {"alt"}, never as
    the empty set. A held binding must therefore compare against the context —
    what else was already down — or 'alt_r' could never match "no modifiers"
    and '<shift>+<alt_r>' could never match "shift".
    """
    own = MODIFIER_MAP.get(key)
    return frozenset(m for m in mods_held if m != own)


def hold_matches(mods_held, key, spec_mods, spec_key) -> bool:
    """True when pressing `key` with `mods_held` satisfies a held binding.

    The binding's modifiers must be a SUBSET of the context, not equal to it.
    Requiring equality looks tidier but is brittle in the real world: any
    stray modifier the OS reports (a key the user is still holding, a
    remapper, a modifier whose release event the tap missed) would silently
    match nothing at all and the hotkey would appear dead. Bindings are tried
    most-specific-first, so matching stays deterministic:

        shift + PTT      -> generate  (shift is held)
        PTT alone        -> dictate   (empty spec, subset of anything)
        cmd + PTT        -> dictate   (as before this feature existed)
    """
    if spec_key is None or not _keys_equal(key, spec_key):
        return False
    return frozenset(spec_mods or ()) <= context_mods(mods_held, key)


def parse_hold_combo(spec: str):
    """'<shift>+<alt_r>' -> (frozenset({'shift'}), Key.alt_r); '' -> (None, None).

    Same grammar as parse_combo, but a HELD binding's trigger may itself be a
    modifier key, so the trigger's own modifier family must not also appear in
    the modifier set — '<alt>+<alt_r>' can never be satisfied.
    """
    mods, trigger = parse_combo(spec)
    if trigger is None:
        return None, None
    own = MODIFIER_MAP.get(trigger)
    if own and own in mods:
        raise ValueError(f"Hold combo modifier collides with its trigger: {spec!r}")
    return mods, trigger


def retarget_hold(spec: str, old_key_name: str, new_key_name: str) -> str:
    """Keep a hold combo pointing at the push-to-talk key when PTT is rebound.

    Returns `spec` unchanged when its trigger is not `old_key_name`, so a
    deliberately independent generation binding is never hijacked.
    """
    if not spec:
        return spec
    try:
        mods, trigger = parse_hold_combo(spec)
    except ValueError:
        return spec
    if trigger is None or key_name(trigger) != old_key_name:
        return spec
    return combo_string(mods, new_key_name)


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
    """Owns the single global listener; dispatches holds, toggle and capture.

    Two HELD bindings share one push-to-talk key: pressing it alone dictates,
    pressing it with the generation combo's modifiers starts a content
    generation. on_ptt_down/on_ptt_up receive the mode ("dictate"|"generate").

    Callbacks fire off the listener thread (see _fire) — they must not touch
    AppKit.
    """

    def __init__(self, ptt_key_name: str, toggle_combo: str,
                 generate_combo: str = "",
                 *, on_ptt_down, on_ptt_up, on_toggle):
        self.on_ptt_down = on_ptt_down
        self.on_ptt_up = on_ptt_up
        self.on_toggle = on_toggle
        self._lock = threading.Lock()
        self._holds = self._build_holds(ptt_key_name, generate_combo)
        # Which hold binding owns the keyboard right now (None = none). The
        # mode is latched at press time, so releasing Shift mid-hold cannot
        # switch a generation session into a dictation one.
        self._hold_key = None
        self._hold_mode = None
        self._combo_mods, self._combo_trigger = parse_combo(toggle_combo)
        self._combo_latched = False
        self._mods_held = set()
        self._capture = None
        # Set while the clarify panel owns the keyboard: the panel is key, so
        # the digits are already going to it — the listener must merely stop
        # holds/toggles from firing underneath the panel.
        self._suppressed = False
        self._listener = None
        # Callbacks are dispatched to this FIFO thread instead of running inline
        # on the pynput tap thread — see _fire.
        self._cb_queue = queue.Queue()
        self._pump = None

    @staticmethod
    def _build_holds(ptt_key_name: str, generate_combo: str) -> dict:
        """Held bindings, MOST SPECIFIC FIRST.

        hold_matches is subset-based, so the empty-modifier `dictate` binding
        matches anything `generate` does; the ordering is what disambiguates
        them and is therefore load-bearing, not cosmetic.
        """
        return {
            "generate": parse_hold_combo(generate_combo),
            "dictate": (frozenset(), resolve_key(ptt_key_name)),
        }

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
            if self._suppressed:
                return
            down_mode = None
            if self._hold_key is None:      # one hold session at a time
                for mode, (spec_mods, spec_key) in self._holds.items():
                    if hold_matches(self._mods_held, key, spec_mods, spec_key):
                        self._hold_key, self._hold_mode = key, mode
                        down_mode = mode
                        break
                else:
                    # A press on a bound key that matched nothing is the
                    # "my hotkey is dead" symptom — record why.
                    if any(_keys_equal(key, k)
                           for _, k in self._holds.values() if k is not None):
                        log("hotkey", f"no hold matched {key!r} "
                                      f"mods={sorted(self._mods_held)} "
                                      f"ctx={sorted(context_mods(self._mods_held, key))}")
            elif _keys_equal(key, self._hold_key):
                pass                        # auto-repeat, expected
            else:
                log("hotkey", f"{key!r} ignored: {self._hold_mode} hold in flight")
            fire_toggle = False
            if (self._combo_trigger is not None
                    and _keys_equal(key, self._combo_trigger)
                    and self._mods_held == self._combo_mods
                    and not self._combo_latched):
                self._combo_latched = True
                fire_toggle = True
        if down_mode:
            self._fire(partial(self.on_ptt_down, down_mode))
        if fire_toggle:
            self._fire(self.on_toggle)

    def _on_release(self, key):
        mod = MODIFIER_MAP.get(key)
        with self._lock:
            if mod:
                self._mods_held.discard(mod)
            if self._combo_latched and (
                    _keys_equal(key, self._combo_trigger) or mod):
                self._combo_latched = False
            # Release matches on key identity ONLY and never re-reads
            # _mods_held: whichever session started is the one that ends.
            up_mode = None
            if self._hold_key is not None and _keys_equal(key, self._hold_key):
                up_mode = self._hold_mode
                self._hold_key = self._hold_mode = None
        if up_mode:
            self._fire(partial(self.on_ptt_up, up_mode))

    # -- off-thread callback dispatch ------------------------------------
    def _fire(self, callback) -> None:
        """Run a PTT/toggle callback OFF the pynput tap thread.

        pynput invokes _on_press/_on_release synchronously on the macOS
        CGEventTap run-loop thread. macOS disables a tap whose callback runs
        longer than ~1 s (kCGEventTapDisabledByTimeout) and pynput 1.8.2 never
        re-enables it. recorder.start()/stop() (opening/closing the audio
        device — slow with Bluetooth mics or under load) could exceed that,
        silently killing the listener: the key release was never delivered, so
        recording never stopped and re-pressing the key did nothing until the
        app was restarted. Dispatching to a dedicated FIFO thread keeps the tap
        callback effectively instant.

        Before the pump is started (unit tests, or pre-start()), run inline so
        callback effects are observable synchronously.
        """
        if self._pump is None:
            callback()
        else:
            self._cb_queue.put(callback)

    def _pump_loop(self) -> None:
        while True:
            callback = self._cb_queue.get()
            if callback is None:  # shutdown sentinel
                return
            try:
                callback()
            except Exception as exc:
                # A failing callback must never kill the dispatch thread, or
                # every later hotkey would be silently dropped.
                log("hotkey", f"callback error: {exc!r}")

    def _start_pump(self) -> None:
        if self._pump is not None:
            return
        self._pump = threading.Thread(
            target=self._pump_loop, name="hotkey-pump", daemon=True
        )
        self._pump.start()

    def _stop_pump(self) -> None:
        if self._pump is None:
            return
        self._pump = None  # _fire falls back to inline once cleared
        self._cb_queue.put(None)  # unblock and end the loop

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

    def set_suppressed(self, on: bool) -> None:
        """Silence hold/toggle dispatch (used while the clarify panel is up).

        Modifier bookkeeping keeps running, so releases stay consistent; only
        the callbacks are withheld. Any in-flight hold is dropped so its
        release cannot fire after the panel closes.
        """
        with self._lock:
            self._suppressed = bool(on)
            if on:
                self._hold_key = self._hold_mode = None
                self._combo_latched = False

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
        self._start_pump()  # before the listener, so no event runs inline
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
        self._stop_pump()

    def update(self, ptt_key_name: str, toggle_combo: str,
               generate_combo: str = "") -> None:
        """Re-bind hotkeys at runtime by swapping match targets. The
        listener is intentionally left untouched (see module docstring)."""
        new_holds = self._build_holds(ptt_key_name, generate_combo)
        new_mods, new_trigger = parse_combo(toggle_combo)
        with self._lock:
            self._holds = new_holds
            self._combo_mods, self._combo_trigger = new_mods, new_trigger
            # Drop any in-flight hold: its key may no longer be bound, and a
            # stuck session would swallow every later press.
            self._hold_key = self._hold_mode = None
            self._combo_latched = False
