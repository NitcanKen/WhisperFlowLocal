"""Launch-at-login via a per-user LaunchAgent plist."""
import os
import plistlib
import subprocess
import sys

from . import BUNDLE_ID, paths


APP_BUNDLE = os.path.expanduser("~/Applications/WhisperFlow-Local.app")


def _plist_content() -> dict:
    if os.path.exists(APP_BUNDLE):
        # Launch through the bundle so macOS attributes permissions
        # (Microphone/Accessibility/Input Monitoring) to the app itself.
        args = ["/usr/bin/open", "-g", APP_BUNDLE]
    else:
        args = [sys.executable, "-m", "whisperflow_local"]
    return {
        "Label": BUNDLE_ID,
        "ProgramArguments": args,
        "WorkingDirectory": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": os.path.join(paths.APP_SUPPORT, "stdout.log"),
        "StandardErrorPath": os.path.join(paths.APP_SUPPORT, "stderr.log"),
    }


def is_enabled() -> bool:
    return os.path.exists(paths.LAUNCH_AGENT)


def enable() -> None:
    os.makedirs(os.path.dirname(paths.LAUNCH_AGENT), exist_ok=True)
    with open(paths.LAUNCH_AGENT, "wb") as f:
        plistlib.dump(_plist_content(), f)
    subprocess.run(
        ["launchctl", "load", "-w", paths.LAUNCH_AGENT],
        capture_output=True, check=False,
    )


def disable() -> None:
    subprocess.run(
        ["launchctl", "unload", "-w", paths.LAUNCH_AGENT],
        capture_output=True, check=False,
    )
    if os.path.exists(paths.LAUNCH_AGENT):
        os.remove(paths.LAUNCH_AGENT)


def toggle() -> bool:
    """Flip launch-at-login; returns the new enabled state."""
    if is_enabled():
        disable()
        return False
    enable()
    return True
