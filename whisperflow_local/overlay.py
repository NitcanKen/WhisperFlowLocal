"""Floating waveform HUD — a Wispr-Flow-style pill above the Dock.

Native look: a borderless, non-activating NSPanel holding an
NSVisualEffectView with the HUD material (real macOS blur/vibrancy),
rounded like a pill, showing a scrolling waveform of live mic levels
while recording and a gentle shimmer while transcribing/formatting.

All methods must be called from the AppKit main thread (the rumps timer).
Pure geometry helpers live at module level so they can be unit-tested.
"""
import math

from AppKit import (
    NSBezierPath,
    NSColor,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
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
    NSBackingStoreBuffered,
)

N_BARS = 36
PILL_W, PILL_H = 300.0, 44.0
BAR_W, BAR_GAP = 3.5, 2.8
BOTTOM_MARGIN = 96.0


# ---------------------------------------------------------- pure helpers

def push_history(history: list, level: float, n: int = N_BARS) -> list:
    """Scroll the waveform: append the newest level, keep the last n."""
    out = list(history) + [max(0.0, min(1.0, float(level)))]
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


# ---------------------------------------------------------- AppKit view

class _WaveView(NSView):
    """Draws the waveform bars and the recording dot."""

    def drawRect_(self, rect):
        heights = getattr(self, "heights", None) or [0.0] * N_BARS
        mode = getattr(self, "mode", "record")
        tick = getattr(self, "tick_count", 0)
        bounds = self.bounds()
        total_w = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP
        left = (bounds.size.width - total_w) / 2.0
        min_h, max_h = 3.0, bounds.size.height - 18.0

        NSColor.whiteColor().colorWithAlphaComponent_(0.88).set()
        for i, lvl in enumerate(heights[:N_BARS]):
            h = min_h + lvl * (max_h - min_h)
            x = left + i * (BAR_W + BAR_GAP)
            y = (bounds.size.height - h) / 2.0
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, BAR_W, h), BAR_W / 2.0, BAR_W / 2.0
            ).fill()

        if mode == "record":
            pulse = 0.55 + 0.45 * (1.0 + math.sin(tick * 0.18)) / 2.0
            NSColor.systemRedColor().colorWithAlphaComponent_(pulse).set()
            d = 8.0
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(16.0, (bounds.size.height - d) / 2.0, d, d)
            ).fill()


# ---------------------------------------------------------- controller

class WaveformOverlay:
    """Owns the HUD panel. Call tick(state, level) ~30x/s from main thread."""

    def __init__(self):
        self._panel = None
        self._wave = None
        self._history = []
        self._t = 0
        self._alpha = 0.0
        self._target = 0.0

    # -- lazy window construction (main thread) -------------------------
    def _build(self):
        screen = NSScreen.mainScreen()
        sf = screen.frame() if screen else NSMakeRect(0, 0, 1440, 900)
        x = sf.origin.x + (sf.size.width - PILL_W) / 2.0
        y = sf.origin.y + BOTTOM_MARGIN
        frame = NSMakeRect(x, y, PILL_W, PILL_H)

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

        content = NSMakeRect(0, 0, PILL_W, PILL_H)
        effect = NSVisualEffectView.alloc().initWithFrame_(content)
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(PILL_H / 2.0)
        effect.layer().setMasksToBounds_(True)

        wave = _WaveView.alloc().initWithFrame_(content)
        wave.heights = [0.0] * N_BARS
        wave.mode = "record"
        wave.tick_count = 0
        effect.addSubview_(wave)

        panel.setContentView_(effect)
        self._panel, self._wave = panel, wave

    # -- per-frame update ------------------------------------------------
    def tick(self, state: str, level: float) -> None:
        """Drive visibility + animation from the app state machine."""
        visible = state in ("recording", "transcribing", "formatting")
        self._target = 1.0 if visible else 0.0

        if visible and self._panel is None:
            self._build()
        if self._panel is None:
            return

        self._t += 1
        if state == "recording":
            self._history = push_history(self._history, level)
            self._wave.mode = "record"
            self._wave.heights = display_heights(self._history)
        elif visible:
            self._wave.mode = "process"
            self._wave.heights = shimmer_heights(N_BARS, float(self._t))
        else:
            self._history = []
        self._wave.tick_count = self._t

        # Smooth fade toward the target alpha (≈120 ms).
        self._alpha += (self._target - self._alpha) * 0.28
        if self._target > 0 and not self._panel.isVisible():
            self._panel.orderFrontRegardless()
        if self._alpha < 0.02 and self._target == 0.0:
            if self._panel.isVisible():
                self._panel.orderOut_(None)
            self._alpha = 0.0
        self._panel.setAlphaValue_(min(1.0, self._alpha))
        if self._panel.isVisible():
            self._wave.setNeedsDisplay_(True)

    def is_visible(self) -> bool:
        return bool(self._panel is not None and self._panel.isVisible())
