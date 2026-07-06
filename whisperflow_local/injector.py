"""Insert text into the frontmost app: clipboard + Cmd+V with clipboard
restore, a direct-typing fallback, and a copy-only mode.

Synthesized keystrokes require the Accessibility permission. When it is
missing, macOS silently drops the Cmd+V event — so insert() checks first
and degrades to leaving the text ON the clipboard (no restore!) and
reporting "clipboard-no-perm" so the UI can tell the user what happened.
"""
import time

import pyperclip
from pynput.keyboard import Controller, Key

from .permissions import accessibility_trusted

_kb = Controller()


def decide_path(copy_only: bool, trusted: bool) -> str:
    """Pure decision used by insert(); unit-tested."""
    if copy_only:
        return "clipboard"
    if not trusted:
        return "clipboard-no-perm"
    return "auto"


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


def type_text(text: str) -> bool:
    """Fallback: synthesize keystrokes directly (slower, but no pasteboard)."""
    try:
        _kb.type(text)
        return True
    except Exception:
        return False


def insert(text: str, copy_only: bool = False) -> str:
    """Insert text using the best available path. Returns the path used:
    "paste" | "type" | "clipboard" | "clipboard-no-perm"."""
    path = decide_path(copy_only, accessibility_trusted())
    if path != "auto":
        # Leave the text on the clipboard — do NOT restore the old contents,
        # otherwise the transcript would be lost entirely.
        copy_to_clipboard(text)
        return path
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
