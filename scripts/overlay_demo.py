"""Visual check for the waveform HUD: shows the real NSPanel for ~7 seconds,
driving bars from the microphone when possible, otherwise from a smooth
sine sweep (clearly reported). Captures a screenshot mid-run.

Usage: .venv/bin/python scripts/overlay_demo.py [screenshot.png]
"""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import objc  # noqa: E402
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory  # noqa: E402
from Foundation import NSObject, NSTimer  # noqa: E402

from whisperflow_local.audio import Recorder  # noqa: E402
from whisperflow_local.overlay import WaveformOverlay  # noqa: E402

SHOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/overlay_demo.png"
DURATION_TICKS = 210  # ~7 s at 30 fps


class Driver(NSObject):
    def initWithOverlay_recorder_(self, overlay, recorder):
        self = objc.super(Driver, self).init()
        self.overlay = overlay
        self.recorder = recorder
        self.ticks = 0
        self.report = []
        return self

    def tick_(self, timer):
        self.ticks += 1
        t = self.ticks / 30.0
        if self.recorder is not None:
            level = self.recorder.level
        else:
            level = 0.15 + 0.55 * abs(math.sin(t * 2.2)) * abs(math.sin(t * 0.7))

        if self.ticks < 140:
            state = "recording"
        elif self.ticks < 190:
            state = "transcribing"  # shimmer mode
        else:
            state = "idle"  # fade out

        self.overlay.tick(state, level)

        if self.ticks == 90:
            mid = (f"mid-run: visible={self.overlay.is_visible()} "
                   f"alpha={self.overlay._panel.alphaValue():.2f} "
                   f"frame={self.overlay._panel.frame()}")
            print(f"[demo] {mid}", flush=True)
            self.report.append(mid)
            # Window-id capture works without full-screen recording rights
            # on some setups; try both, tolerate failure.
            win_id = self.overlay._panel.windowNumber()
            subprocess.run(["screencapture", "-x", "-l", str(win_id), SHOT],
                           check=False)
            if not os.path.exists(SHOT):
                subprocess.run(["screencapture", "-x", SHOT], check=False)
            print(f"[demo] screenshot attempted -> {SHOT} "
                  f"(exists={os.path.exists(SHOT)})", flush=True)
        if self.ticks >= DURATION_TICKS:
            final = (f"final: visible={self.overlay.is_visible()} "
                     f"(expected False after fade-out)")
            print(f"[demo] {final}", flush=True)
            self.report.append(final)
            with open(SHOT + ".txt", "w", encoding="utf-8") as f:
                f.write("\n".join(self.report) + "\n")
            os._exit(0)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    recorder = None
    try:
        recorder = Recorder()
        recorder.start()
        print("[demo] driving bars from REAL microphone input")
    except Exception as exc:
        recorder = None
        print(f"[demo] mic unavailable here ({exc.__class__.__name__}); "
              f"driving bars from a sine sweep")

    overlay = WaveformOverlay()
    driver = Driver.alloc().initWithOverlay_recorder_(overlay, recorder)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / 30.0, driver, "tick:", None, True
    )
    print("[demo] showing HUD pill for ~7s (record -> shimmer -> fade)")
    app.run()


if __name__ == "__main__":
    main()
