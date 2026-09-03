"""Clarify panel — the recording pill grown into a small question window.

Shown when a content-generation request is too vague to write from. It asks
one question at a time (at most two), each with 2-3 options plus a free-text
"Other" field, and is answered by clicking or by pressing 1/2/3.

Native construction mirrors overlay.py and keycap.py (borderless
NSVisualEffectView HUD at NSStatusWindowLevel, same bottom-centre anchor as
the waveform pill) with two deliberate differences:

  * it does NOT setIgnoresMouseEvents_, because it must be clickable, and
  * its NSPanel subclass returns True from canBecomeKeyWindow.

A NONACTIVATING panel that becomes key takes the keyboard WITHOUT activating
the app, so NSWorkspace.frontmostApplication() — and therefore the target of
the ⌘V that pastes the generated text — is unchanged. Verified on this
machine by scripts/itest_clarify_focus.py; re-run it if the panel ever starts
stealing focus.

That also supplies the digit interlock for free, via the responder chain:
while the "Other" field is first responder the digits are text, and only when
it is not does the content view's keyDown_ see them as option shortcuts.

All methods must be called from the AppKit main thread (the rumps timer).
Pure geometry/mapping helpers live at module level so they can be unit-tested.
"""
import queue
import threading
import time

from AppKit import (
    NSAppearance,
    NSAppearanceNameVibrantLight,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSTextField,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)

from .overlay import BOTTOM_MARGIN, PILL_H

CLARIFY_W = 420.0
PAD = 14.0
HEADER_H = 22.0
ROW_H = 30.0
ROW_GAP = 6.0
OTHER_H = 28.0
HINT_H = 16.0
CORNER = 16.0

# The worker blocks while the panel is up, so this is also the longest a
# dictation can sit queued behind it. Longer than the 10 s key-capture
# deadline: reading options and choosing takes longer than pressing a key.
CLARIFY_TIMEOUT = 25.0


# ------------------------------------------------------------ pure helpers

def panel_height(n_options: int) -> float:
    """Total panel height for a question with `n_options` option rows."""
    n = max(0, int(n_options))
    return (PAD + HEADER_H + ROW_GAP
            + n * (ROW_H + ROW_GAP)
            + OTHER_H + ROW_GAP + HINT_H + PAD)


def option_rects(n_options: int, width: float = CLARIFY_W,
                 height: float = None) -> list:
    """Option row frames, top-down, in the panel's flipped-free AppKit coords.

    Row 0 is the topmost (highest y). Every rect is inset by PAD and sits
    strictly inside the panel.
    """
    n = max(0, int(n_options))
    if height is None:
        height = panel_height(n)
    top = height - PAD - HEADER_H - ROW_GAP
    return [
        (PAD, top - (i + 1) * ROW_H - i * ROW_GAP, width - 2 * PAD, ROW_H)
        for i in range(n)
    ]


def other_rect(n_options: int, width: float = CLARIFY_W,
               height: float = None) -> tuple:
    """Frame of the free-text 'Other' field, below the last option row."""
    n = max(0, int(n_options))
    if height is None:
        height = panel_height(n)
    return (PAD, PAD + HINT_H + ROW_GAP, width - 2 * PAD, OTHER_H)


def digit_to_index(ch: str, n_options: int) -> int:
    """'1'..'9' -> a 0-based option index, or -1 when out of range."""
    if not ch or len(ch) != 1 or not ch.isdigit() or ch == "0":
        return -1
    idx = int(ch) - 1
    return idx if 0 <= idx < max(0, int(n_options)) else -1


class ClarifyRequest:
    """One clarify round, handed from the worker thread to the main thread.

    The worker fills `questions` and blocks on `done`; the main thread drives
    the panel and calls resolve() exactly once. `done` supplies the
    happens-before edge, so no extra lock is needed for `answers`.
    """

    def __init__(self, questions: list, timeout: float = CLARIFY_TIMEOUT,
                 clock=time.monotonic):
        self.questions = list(questions or [])
        self.answers = []
        self.state = "pending"        # pending|shown|answered|cancelled|timeout
        self.index = 0                # question currently on screen
        self._clock = clock
        self.deadline = clock() + timeout
        self.timeout = timeout
        self.done = threading.Event()

    def extend_deadline(self) -> None:
        """Give the user a fresh window after each answered question."""
        self.deadline = self._clock() + self.timeout

    def expired(self) -> bool:
        return self._clock() > self.deadline

    def resolve(self, state: str, answers: list = None) -> None:
        """Main thread only. MUST be called only after the panel is hidden —
        the worker synthesizes ⌘V as soon as this returns, and a still-key
        panel would swallow the paste."""
        if self.done.is_set():
            return
        self.state = state
        self.answers = list(answers or [])
        self.done.set()


# ------------------------------------------------------------ AppKit views

class _ClarifyWindow(NSPanel):
    """Nonactivating panel that may still take the keyboard."""

    def canBecomeKeyWindow(self):
        return True


class _ClarifyView(NSView):
    """Hosts the option rows; turns clicks and digits into choices."""

    # `rects` and `choices` are attached by ClarifyPanel after alloc, the
    # same way overlay._WaveView receives `levels`/`mode` — subclassing
    # initWithFrame_ would need objc.super and buys nothing.

    def acceptsFirstResponder(self):
        return True

    def mouseDown_(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        for i, (x, y, w, h) in enumerate(self.rects):
            if x <= p.x <= x + w and y <= p.y <= y + h:
                self.choices.put(f"opt:{i}")
                return

    def keyDown_(self, event):
        # Only reached when the "Other" field is NOT first responder, so a
        # digit typed into that field can never be read as a selection.
        chars = str(event.charactersIgnoringModifiers() or "")
        if event.keyCode() == 53:            # esc
            self.choices.put("cancel")
            return
        idx = digit_to_index(chars[:1], len(self.rects))
        if idx >= 0:
            self.choices.put(f"opt:{idx}")


class ClarifyPanel:
    """Owns the clarify HUD. Main thread only."""

    def __init__(self):
        self._panel = None
        self._view = None
        self._title = None
        self._hint = None
        self._other = None
        self._rows = []
        self._n = 0

    # -- construction -----------------------------------------------------
    def _label(self, size, weight, alpha=1.0):
        field = NSTextField.labelWithString_("")
        field.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
        field.setTextColor_(NSColor.labelColor().colorWithAlphaComponent_(alpha))
        return field

    def _build(self, n_options: int):
        h = panel_height(n_options)
        sf = NSScreen.mainScreen().frame()
        x = sf.origin.x + (sf.size.width - CLARIFY_W) / 2.0
        y = sf.origin.y + BOTTOM_MARGIN
        frame = NSMakeRect(x, y, CLARIFY_W, h)

        panel = _ClarifyWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False,
        )
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        # NOT setIgnoresMouseEvents_: unlike the overlay, this one is clicked.
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )

        content = NSMakeRect(0, 0, CLARIFY_W, h)
        effect = NSVisualEffectView.alloc().initWithFrame_(content)
        effect.setAppearance_(
            NSAppearance.appearanceNamed_(NSAppearanceNameVibrantLight))
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(CORNER)
        effect.layer().setMasksToBounds_(True)

        view = _ClarifyView.alloc().initWithFrame_(content)
        view.rects = []
        view.choices = queue.Queue()
        effect.addSubview_(view)

        title = self._label(14.0, 0.3)
        title.setFrame_(NSMakeRect(PAD, h - PAD - HEADER_H,
                                   CLARIFY_W - 2 * PAD, HEADER_H))
        view.addSubview_(title)

        hint = self._label(11.0, 0.0, alpha=0.55)
        hint.setFrame_(NSMakeRect(PAD, PAD, CLARIFY_W - 2 * PAD, HINT_H))
        view.addSubview_(hint)

        other = NSTextField.alloc().initWithFrame_(
            NSMakeRect(*other_rect(n_options, CLARIFY_W, h)))
        other.setFont_(NSFont.systemFontOfSize_weight_(13.0, 0.0))
        other.setBezeled_(True)
        other.setEditable_(True)
        view.addSubview_(other)

        panel.setContentView_(effect)
        self._panel, self._view, self._title = panel, view, title
        self._hint, self._other, self._n = hint, other, n_options

    # -- lifecycle --------------------------------------------------------
    def show(self, question: str, options: list, hint: str = "") -> None:
        """Display one question. Rebuilds when the option count changes."""
        n = len(options)
        if self._panel is None or n != self._n:
            self.hide()
            self._build(n)
        h = panel_height(n)
        for row in self._rows:
            row.removeFromSuperview()
        self._rows = []
        rects = option_rects(n, CLARIFY_W, h)
        for i, (rect, text) in enumerate(zip(rects, options)):
            row = self._label(13.0, 0.0)
            row.setStringValue_(f"  {i + 1}.  {text}")
            row.setFrame_(NSMakeRect(*rect))
            row.setWantsLayer_(True)
            row.layer().setCornerRadius_(7.0)
            row.layer().setBackgroundColor_(
                NSColor.labelColor().colorWithAlphaComponent_(0.07).CGColor())
            self._view.addSubview_(row)
            self._rows.append(row)
        self._view.rects = rects
        self._title.setStringValue_(question)
        self._hint.setStringValue_(hint)
        self._other.setStringValue_("")
        self._drain()
        self._panel.makeKeyAndOrderFront_(None)
        # Options, not the text field, own the keyboard first — otherwise the
        # digit shortcuts would be typed into "Other" instead.
        self._panel.makeFirstResponder_(self._view)

    def take_choice(self) -> str:
        """Non-blocking: 'opt:N' | 'cancel' | 'other:<text>' | None."""
        if self._panel is None:
            return None
        try:
            return self._view.choices.get_nowait()
        except queue.Empty:
            pass
        text = str(self._other.stringValue() or "").strip()
        if text and self._other_committed():
            return f"other:{text}"
        return None

    def _other_committed(self) -> bool:
        """True once the user has left / submitted the free-text field."""
        return self._panel is not None and not self._is_editing_other()

    def _is_editing_other(self) -> bool:
        responder = self._panel.firstResponder()
        editor = self._other.currentEditor()
        return responder is not None and editor is not None and responder == editor

    def _drain(self) -> None:
        if self._view is None:
            return
        while True:
            try:
                self._view.choices.get_nowait()
            except queue.Empty:
                return

    def hide(self) -> None:
        if self._panel is not None and self._panel.isVisible():
            self._drain()
            self._panel.orderOut_(None)

    def is_visible(self) -> bool:
        return bool(self._panel is not None and self._panel.isVisible())
