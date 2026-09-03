"""Key-capture floating panel — "press the key you want" recorder.

Same native HUD construction as overlay.py (borderless non-activating
NSPanel + NSVisualEffectView), centered on screen. All methods must be
called from the AppKit main thread (the rumps timer). Pure text helpers
live at module level so they can be unit-tested.
"""
from AppKit import (
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSTextAlignmentCenter,
    NSTextField,
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

PANEL_W, PANEL_H = 380.0, 120.0

_MOD_SYMBOL = {"cmd": "⌘", "ctrl": "⌃", "alt": "⌥", "shift": "⇧"}
_KEY_LABEL = {
    "alt": "⌥ Option", "alt_l": "Left ⌥ Option", "alt_r": "Right ⌥ Option",
    "alt_gr": "⌥ AltGr",
    "cmd": "⌘ Command", "cmd_l": "Left ⌘ Command", "cmd_r": "Right ⌘ Command",
    "ctrl": "⌃ Control", "ctrl_l": "Left ⌃ Control", "ctrl_r": "Right ⌃ Control",
    "shift": "⇧ Shift", "shift_l": "Left ⇧ Shift", "shift_r": "Right ⇧ Shift",
    "space": "Space", "tab": "⇥ Tab", "caps_lock": "⇪ Caps Lock",
    "enter": "⏎ Return", "backspace": "⌫ Delete",
}


# ---------------------------------------------------------- pure helpers

def pretty_key(name: str) -> str:
    """Config key name -> human label ('alt_r' -> 'Right ⌥ Option')."""
    if name in _KEY_LABEL:
        return _KEY_LABEL[name]
    return name.upper() if len(name) == 1 else name.upper().replace("_", " ")


def pretty_combo(combo: str) -> str:
    """Combo text -> compact symbols ('<cmd>+<shift>+d' -> '⌘⇧D').

    A held combo's trigger may itself be a named key ('<shift>+<alt_r>'), so
    non-modifier tokens go through pretty_key rather than a bare .upper(),
    which would render 'ALT_R'.
    """
    out = []
    for token in (combo or "").split("+"):
        token = token.strip()
        if not token:
            continue
        inner = token[1:-1] if token.startswith("<") and token.endswith(">") else token
        out.append(_MOD_SYMBOL[inner] if inner in _MOD_SYMBOL else pretty_key(inner))
    return "".join(out)


# ---------------------------------------------------------- controller

class KeyCapturePanel:
    """Owns the capture HUD. show()/show_result()/hide() from main thread."""

    def __init__(self):
        self._panel = None
        self._title = None
        self._body = None

    def _label(self, size, weight, alpha):
        field = NSTextField.labelWithString_("")
        field.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
        field.setTextColor_(NSColor.whiteColor().colorWithAlphaComponent_(alpha))
        field.setAlignment_(NSTextAlignmentCenter)
        return field

    def _build(self):
        screen = NSScreen.mainScreen()
        sf = screen.frame() if screen else NSMakeRect(0, 0, 1440, 900)
        x = sf.origin.x + (sf.size.width - PANEL_W) / 2.0
        y = sf.origin.y + sf.size.height * 0.42
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, PANEL_W, PANEL_H), style, NSBackingStoreBuffered, False
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

        content = NSMakeRect(0, 0, PANEL_W, PANEL_H)
        effect = NSVisualEffectView.alloc().initWithFrame_(content)
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(18.0)
        effect.layer().setMasksToBounds_(True)

        title = self._label(13.0, 0.3, 0.7)
        title.setFrame_(NSMakeRect(12, PANEL_H - 34, PANEL_W - 24, 20))
        body = self._label(22.0, 0.4, 0.95)
        body.setFrame_(NSMakeRect(12, 22, PANEL_W - 24, 46))
        effect.addSubview_(title)
        effect.addSubview_(body)

        panel.setContentView_(effect)
        self._panel, self._title, self._body = panel, title, body

    def show(self, title: str, prompt: str) -> None:
        if self._panel is None:
            self._build()
        self._title.setStringValue_(title)
        self._body.setStringValue_(prompt)
        self._body.setFont_(NSFont.systemFontOfSize_weight_(16.0, 0.2))
        self._panel.orderFrontRegardless()

    def show_result(self, text: str) -> None:
        if self._panel is None:
            return
        self._body.setFont_(NSFont.systemFontOfSize_weight_(22.0, 0.4))
        self._body.setStringValue_(text)

    def hide(self) -> None:
        if self._panel is not None and self._panel.isVisible():
            self._panel.orderOut_(None)

    def is_visible(self) -> bool:
        return bool(self._panel is not None and self._panel.isVisible())
