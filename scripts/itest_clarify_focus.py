"""Focus spike for the clarify panel — the design's one risky assumption.

The clarify panel must accept clicks and typed text WITHOUT activating
WhisperFlow, because injector.insert() pastes into whatever
NSWorkspace.frontmostApplication() is when the answer comes back. If showing
the panel steals the frontmost app, the generated text lands in the wrong
window.

Run it, then look at the printed report:

    .venv/bin/python scripts/itest_clarify_focus.py

PASS  = a borderless non-activating NSPanel can become key while the
        frontmost application is unchanged  -> ClarifyPanel can own an
        editable NSTextField (source "panel").
FAIL  = it cannot  -> fall back to answering through the global listener,
        or by voice.
"""
import sys
import time

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSTextField,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
    NSWorkspace,
)

W, H = 420.0, 120.0


class KeyablePanel(NSPanel):
    def canBecomeKeyWindow(self):
        return True


def front():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(app.localizedName() or "?") if app else "?"


def main() -> int:
    NSApplication.sharedApplication().setActivationPolicy_(
        NSApplicationActivationPolicyAccessory
    )
    before = front()
    print(f"frontmost BEFORE          : {before}")

    sf = NSScreen.mainScreen().frame()
    rect = NSMakeRect(sf.origin.x + (sf.size.width - W) / 2.0,
                      sf.origin.y + 96.0, W, H)
    panel = KeyablePanel.alloc().initWithContentRect_styleMask_backing_defer_(
        rect,
        NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
        NSBackingStoreBuffered, False,
    )
    panel.setLevel_(NSStatusWindowLevel)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.windowBackgroundColor())
    panel.setHasShadow_(True)

    field = NSTextField.alloc().initWithFrame_(NSMakeRect(20, 40, W - 40, 28))
    field.setStringValue_("click me and type")
    panel.contentView().addSubview_(field)

    panel.makeKeyAndOrderFront_(None)
    time.sleep(0.4)

    is_key = bool(panel.isKeyWindow())
    after = front()
    print(f"panel.isKeyWindow()       : {is_key}")
    print(f"frontmost AFTER show      : {after}")

    panel.makeFirstResponder_(field)
    time.sleep(0.4)
    editing = bool(panel.firstResponder() is not None
                   and panel.isKeyWindow())
    after_edit = front()
    print(f"field is first responder  : {editing}")
    print(f"frontmost AFTER focus     : {after_edit}")

    panel.orderOut_(None)
    print(f"frontmost AFTER hide      : {front()}")

    kept_focus = (after == before and after_edit == before)
    print()
    print(f"key window obtained       : {is_key}")
    print(f"frontmost app unchanged   : {kept_focus}")
    verdict = "PASS" if (is_key and kept_focus) else "FAIL"
    print(f"VERDICT                   : {verdict}")
    if verdict == "FAIL":
        print("  -> use the global-listener source (digits) or voice answers,")
        print("     and/or re-activate the captured target app before pasting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
