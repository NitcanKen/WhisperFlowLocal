"""Audible feedback cues using the system sound files (non-blocking)."""
import subprocess

_SOUNDS = {
    "start": "/System/Library/Sounds/Tink.aiff",
    "stop": "/System/Library/Sounds/Pop.aiff",
    "done": "/System/Library/Sounds/Glass.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
}


def play(name: str, enabled: bool = True) -> None:
    if not enabled:
        return
    path = _SOUNDS.get(name)
    if not path:
        return
    try:
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass
