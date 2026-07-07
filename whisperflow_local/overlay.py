"""Floating waveform HUD — a milky-white pill with a woven thread waveform.

Native look: a borderless, non-activating NSPanel holding a light-appearance
NSVisualEffectView tinted cream (米白) to match the app icon, rounded like a
pill. While recording it shows a bundle of hair-thin strands — deep sepia
with champagne-gold accents — that lie as a single quiet line in silence and
fan out and weave around each other as the voice gets louder, tapering to a
point at both ends. While transcribing/formatting the same bundle carries a
gentle travelling wave.

The pill enters like a book opening: it unfolds horizontally from a dot to
full width (0.28 s, easeOutBack overshoot) and folds back shut on dismiss
(0.20 s, easeInCubic). A mid-animation reversal continues from the current
progress — the width factor is blended toward the active curve each frame so
switching direction never jumps.

Rendering is self-driven: the overlay owns a 60 fps NSTimer scheduled in
NSRunLoopCommonModes (so menu tracking never stalls it) and steps all
animation by real elapsed time. `tick(state, level)` only feeds the latest
app state and mic level (~30 Hz from the rumps timer); the frame timer stops
itself once the pill is fully closed.

All methods must be called from the AppKit main thread (the rumps timer).
Pure geometry/easing helpers live at module level so they can be unit-tested.
"""
import math
import time

from AppKit import (
    NSAppearance,
    NSAppearanceNameVibrantLight,
    NSBezierPath,
    NSColor,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
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
    NSBackingStoreBuffered,
)
from Foundation import NSRunLoop, NSRunLoopCommonModes, NSTimer

N_BARS = 36                    # history depth used by legacy helpers/tests
STRAND_POINTS = 24             # samples per strand across the pill
PILL_W, PILL_H = 300.0, 44.0
BOTTOM_MARGIN = 96.0

FRAME_INTERVAL = 1.0 / 60.0    # self-driven render timer
OPEN_DUR = 0.28                # seconds, book unfolds
CLOSE_DUR = 0.20               # seconds, book folds shut
ALPHA_RAMP = 0.30              # alpha reaches 1 over the first 30% of p
BACK_S = 1.1                   # easeOutBack tension → ~4.5% overshoot
WIDTH_BLEND = 0.55             # per-frame blend toward the easing curve
PHASE_RATE = 30.0              # phase units per second (matches old tick=1)

STRAND_LEFT_PAD = 34.0         # clears the recording-dot zone
STRAND_RIGHT_PAD = 16.0
SEPIA_STRANDS = 7
GOLD_STRANDS = 3
STRAND_COUNT = SEPIA_STRANDS + GOLD_STRANDS
LEVEL_FLOOR = 0.05             # bundle stays gently alive in silence
AMP_FRAC = 0.38                # max fan-out as a fraction of pill height
TAPER_EXP = 0.6                # end-taper window sharpness

# Perceptual response: raw mic levels sit low (~0.1–0.3) at normal speaking
# volume, so a linear map barely fans the bundle out. A soft noise gate then
# gain + gamma lifts conversational levels into a strong visual swing while
# true silence (raw <= gate) stays converged.
LEVEL_GATE = 0.04
LEVEL_GAIN = 3.2
LEVEL_GAMMA = 0.55

# Palette from the app icon (sRGB 0..1 triples).
SEPIA = (0.353, 0.275, 0.196)        # #5A4632 — thread body
GOLD_CHAMPAGNE = (0.788, 0.663, 0.416)  # #C9A96A — accent threads
CREAM = (0.949, 0.922, 0.867)        # #F2EBDD — tint over the frosted base
AMBER = (0.831, 0.635, 0.306)        # #D4A24E — recording dot
SEPIA_ALPHA, SEPIA_WIDTH = 0.30, 1.0
GOLD_ALPHA, GOLD_WIDTH = 0.55, 1.2


# ---------------------------------------------------------- pure helpers

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def push_history(history: list, level: float, n: int = N_BARS) -> list:
    """Scroll the waveform: append the newest level, keep the last n."""
    out = list(history) + [clamp01(level)]
    return out[-n:] if len(out) > n else out


def display_heights(history: list, n: int = N_BARS) -> list:
    """Left-pad with zeros so the waveform grows in from the right."""
    h = list(history)[-n:]
    return [0.0] * (n - len(h)) + h


def shimmer_heights(n: int, t: float) -> list:
    """Low-amplitude travelling wave for the processing state (0..1)."""
    return [
        0.10 + 0.10 * (1.0 + math.sin(t * 0.22 + i * 0.55)) / 2.0
        for i in range(n)
    ]


def ease_out_back(p: float, s: float = BACK_S) -> float:
    """Decelerating ease that overshoots 1.0 slightly (~4.5% at s=1.1)."""
    u = clamp01(p) - 1.0
    return 1.0 + (s + 1.0) * u * u * u + s * u * u


def ease_in_cubic(p: float) -> float:
    p = clamp01(p)
    return p * p * p


def unfold_curve(p: float, opening: bool) -> float:
    """Width factor (0..~1.05) along the open or close easing curve."""
    if opening:
        return ease_out_back(p)
    return 1.0 - ease_in_cubic(1.0 - clamp01(p))


def step_width_factor(prev: float, p: float, opening: bool,
                      blend: float = WIDTH_BLEND) -> float:
    """Blend the width factor toward the active curve.

    Direction reversals switch curves; blending instead of snapping keeps
    the pill width continuous (no visible jump) across the switch.
    """
    return prev + (unfold_curve(p, opening) - prev) * blend


def unfold_width(factor: float, w_min: float = PILL_H,
                 w_max: float = PILL_W) -> float:
    """Pill width for a width factor; overshoot may exceed w_max briefly."""
    return w_min + (w_max - w_min) * max(0.0, factor)


def unfold_alpha(p: float) -> float:
    """Fade in over the first ALPHA_RAMP of progress (and out, reversed)."""
    return clamp01(p / ALPHA_RAMP)


def advance_progress(p: float, dt: float, opening: bool,
                     open_dur: float = OPEN_DUR,
                     close_dur: float = CLOSE_DUR) -> float:
    """Step the unfold progress by real elapsed time, clamped to 0..1."""
    if opening:
        return min(1.0, p + dt / open_dur)
    return max(0.0, p - dt / close_dur)


def response_curve(level: float, gain: float = LEVEL_GAIN,
                   gamma: float = LEVEL_GAMMA,
                   gate: float = LEVEL_GATE) -> float:
    """Map a raw mic level to a perceptual fan-out amount (0..1).

    Below the gate → 0 (true silence stays converged); above it, the level
    is rescaled, gained and gamma-lifted so conversational volume produces
    a large, visible swing without needing to shout. Monotonic and bounded.
    """
    x = clamp01(level)
    if x <= gate:
        return 0.0
    x = (x - gate) / (1.0 - gate)
    return clamp01((x * gain) ** gamma)


def smooth_levels(levels: list, window: int = 3) -> list:
    """Centred moving average; preserves length, clamps to 0..1."""
    vals = [clamp01(v) for v in levels]
    if window <= 1 or len(vals) < 2:
        return vals
    half = window // 2
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        seg = vals[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def _frac(x: float) -> float:
    return x - math.floor(x)


def strand_params(k: int) -> tuple:
    """Deterministic personality for strand k: (phase, freq, amp, speed).

    Derived from the index with irrational multipliers so strands stay
    visually distinct and weave across each other — no randomness, so
    every run and every frame sequence is reproducible.
    """
    phase = k * 2.399963  # golden angle, radians
    freq = 0.8 + 0.5 * _frac(k * 0.618034)
    amp = 0.6 + 0.5 * _frac(k * 0.381966)
    speed = 0.7 + 0.6 * _frac(k * 0.5 + 0.25)
    return (phase, freq, amp, speed)


def end_taper(i: int, n: int, exp: float = TAPER_EXP) -> float:
    """0..1 window that pinches the bundle to a point at both ends."""
    if n < 2:
        return 0.0
    return math.sin(math.pi * (i / (n - 1.0))) ** exp


def strand_points(levels: list, t: float, width: float, height: float,
                  amp_scale: float, params: tuple) -> list:
    """Sample one strand of the bundle: [(x, y), ...].

    The envelope (level × end-taper) scales each strand's sine wander, so
    all strands converge onto the centreline in silence and at both ends,
    and fan out / cross while speaking. amp_scale (the unfold progress)
    flattens the whole bundle during the book open/close.
    """
    n = len(levels)
    x0, x1 = STRAND_LEFT_PAD, float(width) - STRAND_RIGHT_PAD
    if n < 2 or (x1 - x0) < 20.0:
        return []
    phase, freq, amp_mult, speed = params
    mid = float(height) / 2.0
    amp = AMP_FRAC * float(height) * clamp01(amp_scale) * amp_mult
    y_lo, y_hi = 2.0, float(height) - 2.0
    pts = []
    for i, raw in enumerate(levels):
        env = max(LEVEL_FLOOR, clamp01(raw)) * end_taper(i, n)
        x = x0 + (x1 - x0) * (i / (n - 1.0))
        y = mid + amp * env * math.sin(0.55 * freq * i + 0.18 * speed * t
                                       + phase)
        pts.append((x, min(y_hi, max(y_lo, y))))
    return pts


def catmull_rom_beziers(points: list) -> list:
    """Cubic-Bezier segments through the points (Catmull-Rom smoothing).

    Returns [(p1, c1, c2, p2), ...] with len(points) - 1 segments; each
    segment's endpoints are consecutive input points.
    """
    if len(points) < 2:
        return []
    ext = [points[0]] + list(points) + [points[-1]]
    segs = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        segs.append((p1, c1, c2, p2))
    return segs


# ---------------------------------------------------------- AppKit view

def _srgb(rgb, alpha=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(
        rgb[0], rgb[1], rgb[2], alpha
    )


def _edge_path(points):
    """Open NSBezierPath smoothly through (x, y) points."""
    path = NSBezierPath.bezierPath()
    segs = catmull_rom_beziers(points)
    if not segs:
        return path
    path.moveToPoint_(segs[0][0])
    for _p1, c1, c2, p2 in segs:
        path.curveToPoint_controlPoint1_controlPoint2_(p2, c1, c2)
    return path


class _WaveView(NSView):
    """Draws the cream tint, the woven strand bundle and the recording dot."""

    def drawRect_(self, rect):
        levels = getattr(self, "levels", None) or []
        mode = getattr(self, "mode", "record")
        phase_t = getattr(self, "phase_t", 0.0)
        amp_scale = getattr(self, "amp_scale", 1.0)
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height

        # Cream tint over the frosted material → 米白, still translucent.
        _srgb(CREAM, 0.55).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, h / 2.0, h / 2.0
        ).fill()

        # Sepia body first, gold accents on top.
        for k in range(STRAND_COUNT):
            pts = strand_points(levels, phase_t, w, h, amp_scale,
                                strand_params(k))
            if not pts:
                break
            path = _edge_path(pts)
            if k < SEPIA_STRANDS:
                path.setLineWidth_(SEPIA_WIDTH)
                _srgb(SEPIA, SEPIA_ALPHA).set()
            else:
                path.setLineWidth_(GOLD_WIDTH)
                _srgb(GOLD_CHAMPAGNE, GOLD_ALPHA).set()
            path.stroke()

        if mode == "record":
            pulse = 0.55 + 0.45 * (1.0 + math.sin(phase_t * 0.18)) / 2.0
            _srgb(AMBER, pulse).set()
            d = 8.0
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(16.0, (h - d) / 2.0, d, d)
            ).fill()


# ---------------------------------------------------------- controller

class WaveformOverlay:
    """Owns the HUD panel and its 60 fps frame timer.

    Call tick(state, level) ~30x/s from the main thread; it feeds the
    latest state and pushes mic levels. All animation happens in the
    self-owned frame timer, which runs only while the pill is on screen.
    """

    def __init__(self):
        self._panel = None
        self._wave = None
        self._history = []
        self._state = "idle"
        self._phase = 0.0      # travelling-wave clock (PHASE_RATE units/s)
        self._p = 0.0          # unfold progress 0..1
        self._wfactor = 0.0    # blended width factor
        self._cx = 0.0         # pill centre x (screen coords)
        self._y = 0.0          # pill bottom y
        self._last_w = -1.0
        self._timer = None
        self._last_ts = None

    # -- lazy window construction (main thread) -------------------------
    def _build(self):
        screen = NSScreen.mainScreen()
        sf = screen.frame() if screen else NSMakeRect(0, 0, 1440, 900)
        self._cx = sf.origin.x + sf.size.width / 2.0
        self._y = sf.origin.y + BOTTOM_MARGIN
        # Born folded: a dot that the first frames unfold into the pill.
        frame = NSMakeRect(self._cx - PILL_H / 2.0, self._y, PILL_H, PILL_H)

        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False
        )
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        panel.setAlphaValue_(0.0)

        content = NSMakeRect(0, 0, PILL_H, PILL_H)
        effect = NSVisualEffectView.alloc().initWithFrame_(content)
        effect.setAppearance_(
            NSAppearance.appearanceNamed_(NSAppearanceNameVibrantLight)
        )
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(PILL_H / 2.0)
        effect.layer().setMasksToBounds_(True)
        effect.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

        wave = _WaveView.alloc().initWithFrame_(content)
        wave.levels = [0.0] * STRAND_POINTS
        wave.mode = "record"
        wave.phase_t = 0.0
        wave.amp_scale = 0.0
        wave.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        effect.addSubview_(wave)

        panel.setContentView_(effect)
        self._panel, self._wave = panel, wave

    # -- frame timer lifecycle -------------------------------------------
    def _start_timer(self):
        if self._timer is not None:
            return
        self._last_ts = time.monotonic()
        timer = NSTimer.timerWithTimeInterval_repeats_block_(
            FRAME_INTERVAL, True, self._on_frame
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(
            timer, NSRunLoopCommonModes
        )
        self._timer = timer

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._last_ts = None

    # -- input feed (~30 Hz, main thread) ---------------------------------
    def tick(self, state: str, level: float) -> None:
        """Feed the app state machine; animation runs on the frame timer."""
        self._state = state
        visible = state in ("recording", "transcribing", "formatting")

        if visible and self._panel is None:
            self._build()
        if self._panel is None:
            return

        if state == "recording":
            self._history = push_history(self._history, level,
                                         n=STRAND_POINTS)
        elif not visible:
            self._history = []

        if visible:
            self._start_timer()

    # -- per-frame update (60 fps, main thread) ----------------------------
    def _on_frame(self, _timer) -> None:
        if self._panel is None:
            self._stop_timer()
            return
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - (self._last_ts or now)))
        self._last_ts = now

        visible = self._state in ("recording", "transcribing", "formatting")
        self._phase += dt * PHASE_RATE

        if self._state == "recording":
            self._wave.mode = "record"
            boosted = [response_curve(v)
                       for v in display_heights(self._history,
                                                n=STRAND_POINTS)]
            self._wave.levels = smooth_levels(boosted)
        elif visible:
            self._wave.mode = "process"
            self._wave.levels = smooth_levels(
                shimmer_heights(STRAND_POINTS, self._phase)
            )
        self._wave.phase_t = self._phase

        # Book unfold/fold: dt-based progress + blended width factor.
        self._p = advance_progress(self._p, dt, visible)
        self._wfactor = step_width_factor(self._wfactor, self._p, visible)
        self._wave.amp_scale = clamp01(self._p)

        if visible and not self._panel.isVisible():
            self._panel.orderFrontRegardless()

        w = unfold_width(self._wfactor)
        if abs(w - self._last_w) > 0.05:
            self._panel.setFrame_display_(
                NSMakeRect(self._cx - w / 2.0, self._y, w, PILL_H), True
            )
            self._last_w = w
        self._panel.setAlphaValue_(unfold_alpha(self._p))

        if not visible and self._p <= 0.0:
            if self._panel.isVisible():
                self._panel.orderOut_(None)
            self._wfactor = 0.0
            self._last_w = -1.0
            self._stop_timer()
            return
        if self._panel.isVisible():
            self._wave.setNeedsDisplay_(True)

    def is_visible(self) -> bool:
        return bool(self._panel is not None and self._panel.isVisible())
