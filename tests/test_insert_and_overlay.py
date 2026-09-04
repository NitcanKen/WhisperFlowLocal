"""Tests for the insert-path decision (Accessibility-aware degradation)
and the waveform HUD's pure geometry helpers."""
import whisperflow_local.injector as injector
from whisperflow_local.injector import TYPING_MAX_SECONDS, decide_path, typing_plan
from whisperflow_local.overlay import (
    N_BARS,
    PILL_H,
    PILL_W,
    STRAND_COUNT,
    STRAND_POINTS,
    advance_progress,
    catmull_rom_beziers,
    display_heights,
    ease_in_cubic,
    ease_out_back,
    end_taper,
    push_history,
    response_curve,
    shimmer_heights,
    smooth_levels,
    step_width_factor,
    strand_params,
    strand_points,
    unfold_alpha,
    unfold_curve,
    unfold_width,
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


# ---------------------------------------------------- typewriter pacing

def _duration(n, **kw):
    chunk, delay = typing_plan(n, **kw)
    return -(-n // chunk) * delay  # bursts * gap


def test_short_text_types_one_char_at_a_time_at_cps():
    chunk, delay = typing_plan(20, cps=60.0)
    assert chunk == 1
    assert abs(delay - 1 / 60.0) < 1e-6


def test_long_text_widens_bursts_instead_of_dragging():
    chunk, _ = typing_plan(600, cps=60.0)
    assert chunk > 1
    # However long the text, the whole insert stays inside the budget.
    for n in (1, 5, 50, 200, 600, 5000):
        assert _duration(n) <= TYPING_MAX_SECONDS + 1e-6


def test_plan_is_empty_for_empty_text():
    assert typing_plan(0) == (0, 0.0)


# ------------------------------------------------ typewriter safety (E1/E3)

def test_typewrite_declines_tabs_and_carriage_returns():
    # Synthesized Tab moves focus and \r is a Return: neither is text, and
    # declining here leaves insert() to paste the text intact instead.
    assert injector.typewrite("a\tb") is False
    assert injector.typewrite("a\rb") is False
    assert injector.typewrite("") is False


def test_typewrite_sends_every_character_in_order(monkeypatch):
    sent = []
    monkeypatch.setattr(injector._kb, "type", sent.append)
    monkeypatch.setattr(injector.time, "sleep", lambda _s: None)
    assert injector.typewrite("你好，world!") is True
    assert "".join(sent) == "你好，world!"


def test_typewrite_pastes_newlines_from_a_pasteboard_staged_once(monkeypatch):
    events = []
    copies = []
    monkeypatch.setattr(injector._kb, "type", lambda t: events.append(t))
    monkeypatch.setattr(injector, "_paste_once", lambda: events.append("\n"))
    monkeypatch.setattr(injector.pyperclip, "paste", lambda: "old")
    monkeypatch.setattr(injector.pyperclip, "copy", copies.append)
    monkeypatch.setattr(injector.time, "sleep", lambda _s: None)
    assert injector.typewrite("ab\ncd") is True
    assert "".join(events) == "ab\ncd"
    # Staged with the newline, never re-staged mid-flight, restored at the end.
    assert copies == ["\n", "old"]


def test_typewrite_finishes_a_failed_insert_instead_of_duplicating(monkeypatch):
    typed = []
    pasted = []

    def explode(piece):
        if len(typed) == 2:
            raise RuntimeError("event tap died")
        typed.append(piece)

    monkeypatch.setattr(injector._kb, "type", explode)
    monkeypatch.setattr(injector, "paste_text",
                        lambda t, restore_clipboard=True: pasted.append(t) or True)
    monkeypatch.setattr(injector.time, "sleep", lambda _s: None)
    assert injector.typewrite("abcdef") is True
    # The remainder is pasted once; insert() must not re-run the whole text.
    assert "".join(typed) + pasted[0] == "abcdef"


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


# ---------------------------------------------------- book-unfold easing

def test_ease_out_back_endpoints_and_overshoot():
    assert abs(ease_out_back(0.0)) < 1e-9
    assert abs(ease_out_back(1.0) - 1.0) < 1e-9
    peak = max(ease_out_back(i / 100.0) for i in range(101))
    assert 1.03 <= peak <= 1.06  # gentle overshoot, not a bounce-fest
    # rises through most of the range before settling back to 1
    assert ease_out_back(0.3) < ease_out_back(0.5) < ease_out_back(0.65)


def test_ease_in_cubic_endpoints_and_shape():
    assert ease_in_cubic(0.0) == 0.0
    assert ease_in_cubic(1.0) == 1.0
    assert ease_in_cubic(0.5) == 0.125  # slow start
    assert ease_in_cubic(-1.0) == 0.0 and ease_in_cubic(2.0) == 1.0


def test_unfold_curve_and_width_endpoints():
    for opening in (True, False):
        assert abs(unfold_curve(0.0, opening)) < 1e-9
        assert abs(unfold_curve(1.0, opening) - 1.0) < 1e-9
    assert unfold_width(0.0) == PILL_H  # folded = a dot
    assert unfold_width(1.0) == PILL_W  # fully open
    assert unfold_width(-0.5) == PILL_H  # never narrower than the dot


def test_unfold_alpha_ramp():
    assert unfold_alpha(0.0) == 0.0
    assert abs(unfold_alpha(0.15) - 0.5) < 1e-9
    assert unfold_alpha(0.3) == 1.0
    assert unfold_alpha(1.0) == 1.0


def test_width_factor_reversal_never_jumps():
    # Open for a few ticks, release mid-animation, then fold shut:
    # the per-tick width change must stay smooth across the reversal.
    factor, p = 0.0, 0.0
    widths = [unfold_width(factor)]
    for _ in range(4):  # opening
        p = min(1.0, p + 1.0 / 9.0)
        factor = step_width_factor(factor, p, True)
        widths.append(unfold_width(factor))
    for _ in range(8):  # released mid-open → folding
        p = max(0.0, p - 1.0 / 6.0)
        factor = step_width_factor(factor, p, False)
        widths.append(unfold_width(factor))
    deltas = [abs(b - a) for a, b in zip(widths, widths[1:])]
    assert max(deltas) < 0.30 * (PILL_W - PILL_H)
    assert widths[-1] < widths[4]  # it does fold back down


# ---------------------------------------------------- dt-based progress

def test_advance_progress_full_open_and_close():
    assert advance_progress(0.0, 0.28, True) == 1.0
    assert advance_progress(1.0, 0.20, False) == 0.0
    assert abs(advance_progress(0.0, 0.14, True) - 0.5) < 1e-9
    assert advance_progress(1.0, 5.0, True) == 1.0   # clamped high
    assert advance_progress(0.0, 5.0, False) == 0.0  # clamped low


# ---------------------------------------------------- thread-bundle strands

def test_smooth_levels_preserves_length_and_bounds():
    levels = [0.0, 1.0, 0.0, 0.8, 0.2]
    out = smooth_levels(levels)
    assert len(out) == len(levels)
    assert all(0.0 <= v <= 1.0 for v in out)
    assert out[1] < 1.0  # the spike is averaged down
    assert smooth_levels([0.4], window=3) == [0.4]
    assert smooth_levels(levels, window=1) == levels


def test_response_curve_boosts_conversational_levels():
    # True silence / noise floor stays converged.
    assert response_curve(0.0) == 0.0
    assert response_curve(0.04) == 0.0
    # Conversational levels get a large lift so no shouting is needed.
    assert response_curve(0.15) > 0.4
    assert response_curve(0.30) > 0.7
    # Monotonic and bounded across the range.
    vals = [response_curve(i / 20.0) for i in range(21)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] == 1.0  # loud saturates


def test_response_curve_makes_bundle_fan_out_at_normal_volume():
    # A moderate mic level (0.2) must produce clearly more spread than raw.
    i_mid = STRAND_POINTS // 2

    def spread(levels):
        ys = [
            strand_points(levels, 2.0, PILL_W, PILL_H, 1.0,
                          strand_params(k))[i_mid][1]
            for k in range(STRAND_COUNT)
        ]
        return max(ys) - min(ys)

    raw = [0.2] * STRAND_POINTS
    boosted = [response_curve(0.2)] * STRAND_POINTS
    assert spread(boosted) > spread(raw) * 1.8  # visibly more open


def test_strand_params_deterministic_and_distinct():
    all_params = [strand_params(k) for k in range(STRAND_COUNT)]
    assert all_params == [strand_params(k) for k in range(STRAND_COUNT)]
    assert len(set(all_params)) == STRAND_COUNT  # every strand has character
    for _phase, freq, amp, speed in all_params:
        assert 0.8 <= freq <= 1.3
        assert 0.6 <= amp <= 1.1
        assert 0.7 <= speed <= 1.3


def test_end_taper_pinches_both_ends():
    n = STRAND_POINTS
    assert end_taper(0, n) == 0.0
    assert end_taper(n - 1, n) < 1e-9
    mid = end_taper((n - 1) // 2, n)
    assert 0.9 <= mid <= 1.0
    assert all(0.0 <= end_taper(i, n) <= 1.0 for i in range(n))
    assert end_taper(0, 1) == 0.0  # degenerate input


def test_strand_points_pinned_to_centreline_at_ends():
    levels = [0.9] * STRAND_POINTS
    mid = PILL_H / 2.0
    for k in range(STRAND_COUNT):
        pts = strand_points(levels, 3.0, PILL_W, PILL_H, 1.0,
                            strand_params(k))
        assert len(pts) == STRAND_POINTS
        xs = [x for x, _y in pts]
        assert xs == sorted(xs) and xs[0] >= 30.0 and xs[-1] <= PILL_W - 10.0
        assert abs(pts[0][1] - mid) < 1e-6    # taper pins the ends
        assert abs(pts[-1][1] - mid) < 1e-6
        assert all(2.0 <= y <= PILL_H - 2.0 for _x, y in pts)


def test_strands_fan_out_with_voice_and_converge_in_silence():
    loud = [0.9] * STRAND_POINTS
    quiet = [0.0] * STRAND_POINTS
    i_mid = STRAND_POINTS // 2

    def spread(levels):
        ys = [
            strand_points(levels, 2.0, PILL_W, PILL_H, 1.0,
                          strand_params(k))[i_mid][1]
            for k in range(STRAND_COUNT)
        ]
        return max(ys) - min(ys)

    assert spread(loud) > spread(quiet) + 2.0  # visibly blooms when speaking
    assert spread(quiet) < 2.0                 # near-single thread in silence


def test_strands_weave_across_each_other():
    levels = [0.9] * STRAND_POINTS
    a = strand_points(levels, 0.0, PILL_W, PILL_H, 1.0, strand_params(0))
    b = strand_points(levels, 0.0, PILL_W, PILL_H, 1.0, strand_params(1))
    diffs = [ya - yb for (_xa, ya), (_xb, yb) in zip(a, b)]
    assert any(d > 0 for d in diffs) and any(d < 0 for d in diffs)


def test_strand_bundle_flattens_when_folded():
    levels = [0.9] * STRAND_POINTS
    mid = PILL_H / 2.0
    for k in range(STRAND_COUNT):
        pts = strand_points(levels, 5.0, PILL_W, PILL_H, 0.0,
                            strand_params(k))
        assert all(abs(y - mid) < 1e-6 for _x, y in pts)


def test_strand_points_empty_when_too_narrow_or_short():
    params = strand_params(0)
    assert strand_points([0.5] * STRAND_POINTS, 0.0, 60.0, PILL_H,
                         1.0, params) == []
    assert strand_points([0.5], 0.0, PILL_W, PILL_H, 1.0, params) == []
    assert strand_points([], 0.0, PILL_W, PILL_H, 1.0, params) == []


def test_catmull_rom_beziers_pass_through_points():
    points = [(0.0, 0.0), (10.0, 5.0), (20.0, -3.0), (30.0, 1.0)]
    segs = catmull_rom_beziers(points)
    assert len(segs) == len(points) - 1
    for (p1, _c1, _c2, p2), a, b in zip(segs, points, points[1:]):
        assert p1 == a and p2 == b
    assert catmull_rom_beziers([(0.0, 0.0)]) == []
    assert catmull_rom_beziers([]) == []
