"""Live macOS permission checks.

Key insight for diagnostics: Input Monitoring alone lets the app SEE the
push-to-talk key, but SYNTHESIZING the Cmd+V keystroke needs Accessibility.
Without it, pasting is silently dropped by macOS — so the app must detect
this and degrade to copy-to-clipboard with a clear notification.
"""

try:
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXIsProcessTrustedWithOptions,
        kAXTrustedCheckOptionPrompt,
    )
    _HAS_AX = True
except ImportError:  # non-macOS or missing binding: assume trusted (fail-open)
    _HAS_AX = False

try:
    from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess
    _HAS_CG = True
except ImportError:
    _HAS_CG = False


def accessibility_trusted() -> bool:
    """True when macOS will accept our synthesized keystrokes (Cmd+V)."""
    if not _HAS_AX:
        return True
    return bool(AXIsProcessTrusted())


def prompt_accessibility() -> bool:
    """Trigger the system Accessibility prompt (adds us to the list)."""
    if not _HAS_AX:
        return True
    return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))


def input_monitoring_granted() -> bool:
    """True when global key listening is permitted."""
    if not _HAS_CG:
        return True
    return bool(CGPreflightListenEventAccess())


def request_input_monitoring() -> None:
    if _HAS_CG:
        CGRequestListenEventAccess()


def summary() -> dict:
    return {
        "accessibility": accessibility_trusted(),
        "input_monitoring": input_monitoring_granted(),
    }


def summary_text() -> str:
    s = summary()
    mark = lambda ok: "✅" if ok else "❌"
    return (
        f"{mark(s['input_monitoring'])} Input Monitoring — global push-to-talk key\n"
        f"{mark(s['accessibility'])} Accessibility — auto-paste (⌘V) into other apps\n"
        f"🎙 Microphone — macOS asks on first recording\n\n"
        + ("" if s["accessibility"] else
           "⚠️ Without Accessibility, text is COPIED to the clipboard instead "
           "of pasted — press ⌘V yourself, or grant Accessibility to your "
           "terminal/Python and restart the app.\n")
    )
