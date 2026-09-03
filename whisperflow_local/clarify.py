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
    NSAttributedString,
    NSLineBreakByTruncatingTail,
    NSMutableParagraphStyle,
    NSBezierPath,
    NSFocusRingTypeNone,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSParagraphStyleAttributeName,
    NSRunLoop,
    NSRunLoopCommonModes,
    NSTimer,
    NSTrackingActiveAlways,
    NSTrackingArea,
    NSTrackingMouseMoved,
    NSTrackingMouseEnteredAndExited,
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
    NSViewHeightSizable,
    NSViewWidthSizable,
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

from .overlay import (
    BOTTOM_MARGIN,
    CREAM,
    GOLD_CHAMPAGNE,
    PILL_H,
    PILL_W,
    SEPIA,
    clamp01,
    unfold_curve,
)

# Geometry. The card is only a little wider than the pill so the morph reads
# as the pill unfolding upward rather than as a second, unrelated window.
CLARIFY_W = 380.0
PAD = 16.0
HEADER_H = 24.0
HEADER_GAP = 12.0
ROW_H = 34.0
ROW_GAP = 7.0
OTHER_H = 30.0
HINT_H = 15.0
CORNER = PILL_H / 2.0      # same radius as the pill: one visual language

# Motion. Slightly slower than the pill's 0.28/0.20 because the card travels
# further; the CURVES are the pill's, so it still reads as the same object.
OPEN_DUR = 0.36
CLOSE_DUR = 0.24
RECT_BLEND = 0.55          # per-frame blend, mirrors overlay.step_width_factor
FRAME_INTERVAL = 1.0 / 60.0

# Content reveal ("print the text elegantly"): elements fade in one after the
# other and rise the last few points into place.
CONTENT_DUR = 0.36         # seconds for the whole staggered reveal
CONTENT_FADE = 0.34        # each element's own fade, as a fraction of it
CONTENT_RISE = 7.0         # points each element travels upward
# The text starts printing while the box is STILL growing. Waiting for the
# box to finish leaves a beat where a full-size empty card just sits there,
# which is exactly what makes a panel feel bolted on rather than unfolded.
CONTENT_GATE = 0.32
CLOSE_GATE_FLOOR = 0.60    # on the way out the text clears first, then the
CLOSE_GATE_FADE = 0.30     # empty box folds back down to the pill

# The worker blocks while the panel is up, so this is also the longest a
# dictation can sit queued behind it. Longer than the 10 s key-capture
# deadline: reading options and choosing takes longer than pressing a key.
CLARIFY_TIMEOUT = 25.0


# ------------------------------------------------------------ pure helpers
# Laid out BOTTOM-UP. The card's bottom edge is pinned to the pill's, so
# bottom-anchored positions stay put while the height animates — the content
# is revealed in place instead of sliding around as the box grows.

def _rows_bottom() -> float:
    return PAD + HINT_H + ROW_GAP + OTHER_H + ROW_GAP


def rows_top(n_options: int) -> float:
    n = max(0, int(n_options))
    if n == 0:
        return _rows_bottom()
    return _rows_bottom() + n * ROW_H + (n - 1) * ROW_GAP


def panel_height(n_options: int) -> float:
    """Total card height for a question with `n_options` option rows."""
    return rows_top(n_options) + HEADER_GAP + HEADER_H + PAD


def option_rects(n_options: int, width: float = CLARIFY_W,
                 height: float = None) -> list:
    """Option row frames, row 0 topmost, inset by PAD."""
    n = max(0, int(n_options))
    base = _rows_bottom()
    return [
        (PAD, base + (n - 1 - i) * (ROW_H + ROW_GAP), width - 2 * PAD, ROW_H)
        for i in range(n)
    ]


def other_rect(n_options: int = 0, width: float = CLARIFY_W,
               height: float = None) -> tuple:
    """Frame of the free-text field, below the last option row."""
    return (PAD, PAD + HINT_H + ROW_GAP, width - 2 * PAD, OTHER_H)


def title_rect(n_options: int, width: float = CLARIFY_W) -> tuple:
    return (PAD, rows_top(n_options) + HEADER_GAP, width - 2 * PAD, HEADER_H)


def hint_rect(width: float = CLARIFY_W) -> tuple:
    return (PAD, PAD, width - 2 * PAD, HINT_H)


def digit_to_index(ch: str, n_options: int) -> int:
    """'1'..'9' -> a 0-based option index, or -1 when out of range."""
    if not ch or len(ch) != 1 or not ch.isdigit() or ch == "0":
        return -1
    idx = int(ch) - 1
    return idx if 0 <= idx < max(0, int(n_options)) else -1


def morph_rect(p: float, opening: bool, start: tuple, end: tuple) -> tuple:
    """Interpolate the pill rect toward the card rect along the PILL's own
    easing curve, so opening keeps the same slight overshoot."""
    f = unfold_curve(p, opening)
    return tuple(a + (b - a) * f for a, b in zip(start, end))


def blend_rect(prev: tuple, want: tuple, blend: float = RECT_BLEND) -> tuple:
    """Ease the live rect toward the wanted one. Also absorbs a mid-flight
    target change (question 2 having a different option count) without a jump."""
    if prev is None:
        return want
    return tuple(a + (b - a) * blend for a, b in zip(prev, want))


def content_alpha(cp: float, index: int, count: int,
                  fade: float = CONTENT_FADE) -> float:
    """Per-element opacity for the staggered reveal, over content progress."""
    count = max(1, int(count))
    span = max(0.0, 1.0 - fade)
    step = span / (count - 1) if count > 1 else 0.0
    return clamp01((clamp01(cp) - index * step) / fade)


def close_gate(p: float, opening: bool) -> float:
    """Multiplier that empties the card before it shrinks, and holds the text
    hidden until it has nearly finished opening."""
    if opening:
        return clamp01((p - CONTENT_GATE) / (1.0 - CONTENT_GATE))
    return clamp01((p - CLOSE_GATE_FLOOR) / CLOSE_GATE_FADE)


def content_offset(alpha: float, rise: float = CONTENT_RISE) -> float:
    """Points an element is still below its final position."""
    return (1.0 - clamp01(alpha)) * rise
def _srgb(rgb, alpha=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(
        rgb[0], rgb[1], rgb[2], alpha)


def _text(s: str, size: float, weight: float, rgb, alpha: float,
          right: bool = False):
    """One-line attributed string in the app-icon palette."""
    style = NSMutableParagraphStyle.alloc().init()
    style.setLineBreakMode_(NSLineBreakByTruncatingTail)
    if right:
        style.setAlignment_(2)  # NSTextAlignmentRight
    return NSAttributedString.alloc().initWithString_attributes_(str(s), {
        NSFontAttributeName: NSFont.systemFontOfSize_weight_(size, weight),
        NSForegroundColorAttributeName: _srgb(rgb, alpha),
        NSParagraphStyleAttributeName: style,
    })


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
        """Main thread only. MUST be called only after the panel has finished
        closing — the worker synthesizes Cmd+V as soon as this returns."""
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
    """Draws the card in the app-icon palette and turns clicks into choices.

    Everything except the free-text field is drawn here rather than built from
    subviews, so each element's opacity and its few points of upward travel
    can be driven per frame for the staggered reveal.

    Attributes are attached by ClarifyPanel after alloc (the same pattern as
    overlay._WaveView) — subclassing initWithFrame_ would need objc.super and
    buys nothing.
    """

    def acceptsFirstResponder(self):
        return True

    def isFlipped(self):
        return False

    # -- hit testing ------------------------------------------------------
    def _row_at(self, point):
        for i, (x, y, w, h) in enumerate(getattr(self, "rects", [])):
            if x <= point.x <= x + w and y <= point.y <= y + h:
                return i
        return -1

    def mouseDown_(self, event):
        if not getattr(self, "interactive", True):
            return
        idx = self._row_at(self.convertPoint_fromView_(
            event.locationInWindow(), None))
        if idx >= 0:
            self.choices.put(f"opt:{idx}")

    def mouseMoved_(self, event):
        idx = self._row_at(self.convertPoint_fromView_(
            event.locationInWindow(), None))
        if idx != getattr(self, "hover", -1):
            self.hover = idx
            self.setNeedsDisplay_(True)

    def mouseExited_(self, event):
        if getattr(self, "hover", -1) != -1:
            self.hover = -1
            self.setNeedsDisplay_(True)

    def updateTrackingAreas(self):
        for area in list(self.trackingAreas()):
            self.removeTrackingArea_(area)
        self.addTrackingArea_(
            NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                self.bounds(),
                NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited
                | NSTrackingActiveAlways,
                self, None))

    def keyDown_(self, event):
        # Only reached when the free-text field is NOT first responder, so a
        # digit typed into that field can never be read as a selection.
        if not getattr(self, "interactive", True):
            return
        if event.keyCode() == 53:            # esc
            self.choices.put("cancel")
            return
        chars = str(event.charactersIgnoringModifiers() or "")
        idx = digit_to_index(chars[:1], len(getattr(self, "rects", [])))
        if idx >= 0:
            self.choices.put(f"opt:{idx}")

    # -- drawing ----------------------------------------------------------
    def drawRect_(self, rect):
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height
        gate = getattr(self, "gate", 0.0)
        cp = getattr(self, "content_p", 0.0)
        title = getattr(self, "title_text", "")
        options = getattr(self, "options", [])
        hint = getattr(self, "hint_text", "")
        hover = getattr(self, "hover", -1)
        n = len(options)
        count = n + 3                        # title, rows..., field, hint

        # Cream tint over the frosted material — identical to the pill, so the
        # card reads as the same material simply grown larger.
        _srgb(CREAM, 0.55).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, CORNER, CORNER).fill()
        if h < PILL_H + 2.0 or gate <= 0.0:
            return                           # still essentially the pill

        def alpha_for(i):
            return content_alpha(cp, i, count) * gate

        a = alpha_for(0)
        if a > 0.01 and title:
            tx, ty, tw, th = title_rect(n, w)
            _text(title, 15.0, 0.35, SEPIA, a).drawInRect_(
                NSMakeRect(tx, ty - content_offset(a), tw, th))

        for i, (label, (rx, ry, rw, rh)) in enumerate(
                zip(options, option_rects(n, w))):
            a = alpha_for(1 + i)
            if a <= 0.01:
                continue
            dy = content_offset(a)
            row = NSMakeRect(rx, ry - dy, rw, rh)
            lift = 0.30 if i == hover else 0.16
            _srgb(GOLD_CHAMPAGNE, lift * a).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                row, 9.0, 9.0).fill()
            if i == hover:
                _srgb(GOLD_CHAMPAGNE, 0.55 * a).set()
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    row, 9.0, 9.0)
                path.setLineWidth_(1.0)
                path.stroke()
            baseline = row.origin.y + (rh - 19.0) / 2.0
            _text(f"{i + 1}", 13.0, 0.5, GOLD_CHAMPAGNE, 0.95 * a).drawInRect_(
                NSMakeRect(rx + 14.0, baseline, 16.0, 19.0))
            _text(label, 14.0, 0.0, SEPIA, 0.92 * a).drawInRect_(
                NSMakeRect(rx + 34.0, baseline, rw - 46.0, 19.0))

        a = alpha_for(count - 2)
        if a > 0.01:
            ox, oy, ow, oh = other_rect(n, w)
            dy = content_offset(a)
            field = NSMakeRect(ox, oy - dy, ow, oh)
            _srgb(SEPIA, 0.05 * a).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                field, 8.0, 8.0).fill()
            if getattr(self, "other_empty", True):
                _text(getattr(self, "other_prompt", ""), 12.5, 0.0,
                      SEPIA, 0.34 * a).drawInRect_(
                          NSMakeRect(ox + 10.0, field.origin.y + (oh - 17.0) / 2.0,
                                     ow - 20.0, 17.0))

        a = alpha_for(count - 1)
        if a > 0.01 and hint:
            hx, hy, hw, hh = hint_rect(w)
            _text(hint, 11.0, 0.0, SEPIA, 0.45 * a).drawInRect_(
                NSMakeRect(hx, hy - content_offset(a), hw, hh))


# ---------------------------------------------------------- controller

class ClarifyPanel:
    """Owns the clarify card and its 60 fps frame timer. Main thread only.

    The card is born at EXACTLY the waveform pill's rect and grows from it
    along the pill's own easing curve, so the two read as one object: the
    pill unfolds upward into a card, the text prints itself in, and on the
    way out the text leaves first and the card folds back down to the pill.
    """

    def __init__(self):
        self._panel = None
        self._view = None
        self._other = None
        self._timer = None
        self._last_ts = None
        self._p = 0.0            # 0 = pill-sized, 1 = fully open
        self._opening = False
        self._content_p = 0.0
        self._rect = None
        self._n = 0

    # -- geometry ---------------------------------------------------------
    def _anchor(self):
        screen = NSScreen.mainScreen()
        sf = screen.frame() if screen else NSMakeRect(0, 0, 1440, 900)
        cx = sf.origin.x + sf.size.width / 2.0
        return cx, sf.origin.y + BOTTOM_MARGIN

    def _pill_rect(self):
        cx, y = self._anchor()
        return (cx - PILL_W / 2.0, y, PILL_W, PILL_H)

    def _card_rect(self):
        cx, y = self._anchor()
        return (cx - CLARIFY_W / 2.0, y, CLARIFY_W, panel_height(self._n))

    # -- construction -----------------------------------------------------
    def _build(self):
        rect = self._pill_rect()
        panel = _ClarifyWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(*rect),
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
        panel.setAlphaValue_(0.0)

        content = NSMakeRect(0, 0, rect[2], rect[3])
        effect = NSVisualEffectView.alloc().initWithFrame_(content)
        effect.setAppearance_(
            NSAppearance.appearanceNamed_(NSAppearanceNameVibrantLight))
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(CORNER)
        effect.layer().setMasksToBounds_(True)
        effect.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

        view = _ClarifyView.alloc().initWithFrame_(content)
        view.rects = []
        view.choices = queue.Queue()
        view.hover = -1
        view.gate = 0.0
        view.content_p = 0.0
        view.interactive = False
        view.title_text = ""
        view.options = []
        view.hint_text = ""
        view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        effect.addSubview_(view)

        other = NSTextField.alloc().initWithFrame_(NSMakeRect(*other_rect()))
        other.setFont_(NSFont.systemFontOfSize_weight_(13.0, 0.0))
        other.setBezeled_(False)
        other.setBordered_(False)
        # The view draws the field's ground (and its prompt when empty) so an
        # empty field does not read as a dead third option row.
        other.setDrawsBackground_(False)
        other.setTextColor_(_srgb(SEPIA, 0.92))
        other.setFocusRingType_(NSFocusRingTypeNone)
        other.setEditable_(True)
        other.setAlphaValue_(0.0)
        view.addSubview_(other)

        panel.setContentView_(effect)
        self._panel, self._view, self._other = panel, view, other
        self._rect = rect

    # -- frame timer ------------------------------------------------------
    def _start_timer(self):
        if self._timer is not None:
            return
        self._last_ts = time.monotonic()
        timer = NSTimer.timerWithTimeInterval_repeats_block_(
            FRAME_INTERVAL, True, self._on_frame)
        NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
        self._timer = timer

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._last_ts = None

    def _on_frame(self, _timer):
        if self._panel is None:
            self._stop_timer()
            return
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - (self._last_ts or now)))
        self._last_ts = now

        self._p = (min(1.0, self._p + dt / OPEN_DUR) if self._opening
                   else max(0.0, self._p - dt / CLOSE_DUR))
        gate = close_gate(self._p, self._opening)
        if self._opening and gate > 0.0:
            self._content_p = min(1.0, self._content_p + dt / CONTENT_DUR)

        want = morph_rect(self._p, self._opening,
                          self._pill_rect(), self._card_rect())
        self._rect = blend_rect(self._rect, want)
        self._panel.setFrame_display_(NSMakeRect(*self._rect), True)
        self._panel.setAlphaValue_(clamp01(self._p / 0.25))

        w = self._rect[2]
        self._view.gate = gate
        self._view.content_p = self._content_p
        self._view.other_empty = not str(
            self._other.stringValue() or "").strip()
        self._view.rects = option_rects(self._n, w)
        self._view.interactive = self._opening and gate > 0.6
        self._view.setNeedsDisplay_(True)

        ox, oy, ow, oh = other_rect(self._n, w)
        field_a = content_alpha(self._content_p, self._n + 1,
                                self._n + 3) * gate
        self._other.setFrame_(
            NSMakeRect(ox, oy - content_offset(field_a), ow, oh))
        self._other.setAlphaValue_(field_a)
        self._other.setEditable_(self._view.interactive)

        if not self._opening and self._p <= 0.0:
            self._panel.orderOut_(None)
            self._rect = self._pill_rect()
            self._stop_timer()

    # -- lifecycle --------------------------------------------------------
    def show(self, question: str, options: list, hint: str = "",
             other_prompt: str = "") -> None:
        """Open the card on one question, or swap the question already shown.

        Swapping keeps the card on screen and simply replays the staggered
        reveal; blend_rect absorbs the height change when the new question
        has a different number of options.
        """
        if self._panel is None:
            self._build()
        first = not self._opening
        self._n = len(options)
        self._view.title_text = question
        self._view.options = list(options)
        self._view.hint_text = hint
        self._view.other_prompt = other_prompt
        self._view.other_empty = True
        self._view.hover = -1
        self._view.rects = option_rects(self._n, self._rect[2])
        self._content_p = 0.0
        self._opening = True
        self._other.setStringValue_("")
        self._panel.setIgnoresMouseEvents_(False)
        self._drain()
        if first:
            self._rect = self._pill_rect()
            self._panel.setFrame_display_(NSMakeRect(*self._rect), True)
            self._panel.setAlphaValue_(0.0)
            self._panel.makeKeyAndOrderFront_(None)
        # Options own the keyboard first; the free-text field takes it only
        # when clicked, which is what keeps digits from landing in it.
        self._panel.makeFirstResponder_(self._view)
        self._start_timer()

    def begin_hide(self) -> None:
        """Start the closing animation. is_visible() stays True until it has
        finished, so the caller can wait before pasting."""
        if self._panel is None or not self._opening:
            return
        self._opening = False
        self._view.interactive = False
        self._other.setEditable_(False)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.makeFirstResponder_(self._view)
        self._start_timer()

    def take_choice(self) -> str:
        """Non-blocking: 'opt:N' | 'cancel' | 'other:<text>' | None."""
        if self._panel is None:
            return None
        try:
            return self._view.choices.get_nowait()
        except queue.Empty:
            pass
        if not self._view.interactive:
            return None
        text = str(self._other.stringValue() or "").strip()
        if text and not self._is_editing_other():
            return f"other:{text}"
        return None

    def _is_editing_other(self) -> bool:
        responder = self._panel.firstResponder()
        editor = self._other.currentEditor()
        return responder is not None and editor is not None and responder == editor

    def _drain(self) -> None:
        while True:
            try:
                self._view.choices.get_nowait()
            except queue.Empty:
                return

    def hide(self) -> None:
        """Immediate teardown (error paths); prefer begin_hide()."""
        self._opening = False
        self._p = 0.0
        if self._panel is not None and self._panel.isVisible():
            self._drain()
            self._panel.orderOut_(None)
        self._stop_timer()

    def is_visible(self) -> bool:
        return bool(self._panel is not None and self._panel.isVisible())
