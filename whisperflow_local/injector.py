"""Insert text into the frontmost app: clipboard + Cmd+V with clipboard
restore, a typewriter path, a direct-typing fallback, and a copy-only mode.

Synthesized keystrokes require the Accessibility permission. When it is
missing, macOS silently drops the Cmd+V event — so insert() checks first
and degrades to leaving the text ON the clipboard (no restore!) and
reporting "clipboard-no-perm" so the UI can tell the user what happened.
"""
import math
import time

import pyperclip
from pynput.keyboard import Controller, Key

from .permissions import accessibility_trusted

_kb = Controller()

# Typewriter pacing. The text is already final when we start — this is purely
# the *feel* of it landing, so it is capped hard: a long paragraph grows the
# per-tick chunk instead of dragging the user through a slow crawl.
TYPING_CPS = 60.0          # characters/second for short utterances
TYPING_TICK = 0.012        # target gap between bursts (~83 Hz reads continuous)
TYPING_MAX_SECONDS = 1.1   # whole-insert budget, however long the text is
# Synthesized Return/Tab are NOT text: Return sends the message in Slack and
# Messages, Tab moves focus. Newlines therefore ride the pasteboard, and text
# carrying the rarer two skips the effect entirely rather than misfire.
_UNTYPABLE = "\r\t"


def decide_path(copy_only: bool, trusted: bool) -> str:
    """Pure decision used by insert(); unit-tested."""
    if copy_only:
        return "clipboard"
    if not trusted:
        return "clipboard-no-perm"
    return "auto"


def typing_plan(n: int, cps: float = TYPING_CPS, tick: float = TYPING_TICK,
                max_seconds: float = TYPING_MAX_SECONDS) -> tuple:
    """(chars_per_burst, delay_between_bursts) for typing `n` characters.

    Short text types at `cps`, one character at a time. Past the budget the
    bursts widen rather than the animation lengthening, so the total stays
    under `max_seconds` for any length. Pure, so it is unit-tested.
    """
    if n <= 0:
        return 0, 0.0
    budget = min(n / max(cps, 1e-6), max_seconds)
    ticks = max(1, round(budget / tick))
    chunk = max(1, math.ceil(n / ticks))
    return chunk, budget / math.ceil(n / chunk)


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


def paste_text(text: str, restore_clipboard: bool = True) -> bool:
    """Paste `text` into the focused app via clipboard + synthesized Cmd+V.

    Returns True on success. The previous clipboard contents are restored
    afterwards. Requires the Accessibility permission.
    """
    previous = None
    if restore_clipboard:
        try:
            previous = pyperclip.paste()
        except pyperclip.PyperclipException:
            previous = None
    try:
        pyperclip.copy(text)
        time.sleep(0.08)  # let the pasteboard settle
        with _kb.pressed(Key.cmd):
            _kb.press("v")
            _kb.release("v")
        time.sleep(0.30)  # give the target app time to read the pasteboard
        return True
    except Exception:
        return False
    finally:
        if restore_clipboard and previous is not None:
            try:
                pyperclip.copy(previous)
            except pyperclip.PyperclipException:
                pass


def _paste_once() -> None:
    with _kb.pressed(Key.cmd):
        _kb.press("v")
        _kb.release("v")


def typewrite(text: str, cps: float = TYPING_CPS,
              restore_clipboard: bool = True) -> bool:
    """Insert `text` as a fast burst of keystrokes, so it lands the way typing
    does instead of appearing all at once.

    Unlike chunked pasting there is no pasteboard race: every character rides
    inside its own key event, and the events are consumed in the order they
    were posted. Newlines are the exception — they are pasted from a pasteboard
    staged ONCE with "\n" and left untouched for the whole animation, so the
    target app can only ever read the newline we meant.

    Returns False without typing anything when the effect can't be delivered
    safely, leaving insert() to fall back to a plain paste.
    """
    if not text or any(c in text for c in _UNTYPABLE):
        return False
    chunk, delay = typing_plan(len(text), cps=cps)
    previous = None
    staged = False
    if "\n" in text:
        if restore_clipboard:
            try:
                previous = pyperclip.paste()
            except pyperclip.PyperclipException:
                previous = None
        try:
            pyperclip.copy("\n")
        except pyperclip.PyperclipException:
            return False
        staged = True
        time.sleep(0.08)  # let the pasteboard settle before the first ⌘V
    typed = 0
    try:
        while typed < len(text):
            if text[typed] == "\n":
                _paste_once()
                typed += 1
                time.sleep(max(delay, 0.04))
                continue
            stop = text.find("\n", typed, typed + chunk)
            stop = typed + chunk if stop == -1 else stop
            piece = text[typed:stop]
            _kb.type(piece)
            typed += len(piece)
            if typed < len(text):
                time.sleep(delay)
        return True
    except Exception:
        # Never let a half-typed insert be retried as a whole one — that would
        # duplicate the text. Finish the remainder here, or report progress.
        rest = text[typed:]
        if not rest:
            return True
        if paste_text(rest, restore_clipboard=False):
            return True
        return typed > 0
    finally:
        if staged and previous is not None:
            try:
                pyperclip.copy(previous)
            except pyperclip.PyperclipException:
                pass


def type_text(text: str) -> bool:
    """Fallback: synthesize keystrokes directly (slower, but no pasteboard)."""
    try:
        _kb.type(text)
        return True
    except Exception:
        return False


def insert(text: str, copy_only: bool = False, typing: bool = False,
           cps: float = TYPING_CPS) -> str:
    """Insert text using the best available path. Returns the path used:
    "paste" | "type" | "clipboard" | "clipboard-no-perm"."""
    path = decide_path(copy_only, accessibility_trusted())
    if path != "auto":
        # Leave the text on the clipboard — do NOT restore the old contents,
        # otherwise the transcript would be lost entirely.
        copy_to_clipboard(text)
        return path
    if typing and typewrite(text, cps=cps):
        return "type"
    if paste_text(text):
        return "paste"
    if type_text(text):
        return "type"
    copy_to_clipboard(text)
    return "clipboard"


def press_enter() -> None:
    _kb.press(Key.enter)
    _kb.release(Key.enter)


def delete_chars(count: int) -> None:
    """Send backspaces (used by 'scratch that' to undo the last insertion)."""
    for _ in range(max(0, count)):
        _kb.press(Key.backspace)
        _kb.release(Key.backspace)
        time.sleep(0.005)
