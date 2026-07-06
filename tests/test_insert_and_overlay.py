"""Tests for the insert-path decision (Accessibility-aware degradation)
and the waveform HUD's pure geometry helpers."""
from whisperflow_local.injector import decide_path
from whisperflow_local.overlay import (
    N_BARS,
    display_heights,
    push_history,
    shimmer_heights,
)

# ---------------------------------------------------- insert path (E1/E3/H2)

def test_copy_only_always_clipboard():
    assert decide_path(copy_only=True, trusted=True) == "clipboard"
    assert decide_path(copy_only=True, trusted=False) == "clipboard"


def test_no_accessibility_degrades_to_clipboard_no_perm():
    # The bug from the field: keystrokes silently dropped without the
    # Accessibility permission. Must degrade, never lose the transcript.
    assert decide_path(copy_only=False, trusted=False) == "clipboard-no-perm"


def test_trusted_uses_auto_paste():
    assert decide_path(copy_only=False, trusted=True) == "auto"


# ---------------------------------------------------- waveform helpers (A4)

def test_push_history_scrolls_and_clamps():
    h = []
    for lvl in [0.2, 0.5, 1.7, -0.3]:
        h = push_history(h, lvl, n=3)
    assert len(h) == 3
    assert h == [0.5, 1.0, 0.0]  # clamped to 0..1, oldest dropped


def test_display_heights_pads_left():
    out = display_heights([0.4, 0.6], n=5)
    assert out == [0.0, 0.0, 0.0, 0.4, 0.6]
    assert len(display_heights([], n=N_BARS)) == N_BARS


def test_shimmer_heights_bounded_and_animated():
    a = shimmer_heights(N_BARS, t=0.0)
    b = shimmer_heights(N_BARS, t=10.0)
    assert len(a) == N_BARS
    assert all(0.0 <= v <= 0.25 for v in a)  # gentle, never full-height
    assert a != b  # animates over time
