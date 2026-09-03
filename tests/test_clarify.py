"""Clarify panel: geometry, key mapping, and the worker/main-thread handoff.

Only the pure half is tested — no NSPanel is created, the same way
tests/test_insert_and_overlay.py covers overlay.py's module-level helpers.
"""
import threading

from whisperflow_local.clarify import (
    CLARIFY_TIMEOUT,
    CLARIFY_W,
    PAD,
    ClarifyRequest,
    digit_to_index,
    option_rects,
    other_rect,
    panel_height,
)
from whisperflow_local.llm import (
    MAX_CLARIFY_OPTIONS,
    MAX_CLARIFY_QUESTIONS,
    parse_clarify,
)

QS = [{"question": "用咩語言寫？", "options": ["English", "繁體中文"]}]


# ------------------------------------------------------------- geometry

def test_panel_height_grows_with_the_option_count():
    assert panel_height(3) > panel_height(2) > panel_height(0)


def test_option_rects_are_stacked_top_down_inside_the_panel():
    h = panel_height(3)
    rects = option_rects(3, CLARIFY_W, h)
    assert len(rects) == 3
    ys = [y for _, y, _, _ in rects]
    assert ys == sorted(ys, reverse=True)          # row 0 is topmost
    for x, y, w, rh in rects:
        assert x >= PAD and y >= 0 and x + w <= CLARIFY_W and y + rh <= h


def test_option_rows_do_not_overlap_each_other():
    rects = option_rects(3)
    for (_, y1, _, h1), (_, y2, _, _) in zip(rects, rects[1:]):
        assert y2 + rects[1][3] <= y1          # next row sits fully below


def test_other_field_sits_below_every_option_row():
    n = 3
    h = panel_height(n)
    _, oy, _, oh = other_rect(n, CLARIFY_W, h)
    lowest = min(y for _, y, _, _ in option_rects(n, CLARIFY_W, h))
    assert oy + oh <= lowest
    assert oy >= 0


def test_zero_options_still_produces_a_sane_panel():
    assert option_rects(0) == []
    assert panel_height(0) > 0


# ------------------------------------------------------------- key mapping

def test_digit_to_index_maps_one_based_digits():
    assert digit_to_index("1", 3) == 0
    assert digit_to_index("3", 3) == 2


def test_digit_to_index_rejects_out_of_range_and_non_digits():
    for ch in ("4", "0", "a", "", "12", None):
        assert digit_to_index(ch, 3) == -1


# --------------------------------------------------- worker/main handoff

def test_resolve_publishes_answers_across_threads():
    req = ClarifyRequest(QS)
    seen = {}

    def worker():
        req.done.wait(2.0)
        seen["state"] = req.state
        seen["answers"] = list(req.answers)

    t = threading.Thread(target=worker)
    t.start()
    req.resolve("answered", ["English"])
    t.join(2.0)
    assert not t.is_alive()
    assert seen == {"state": "answered", "answers": ["English"]}


def test_resolve_is_idempotent():
    req = ClarifyRequest(QS)
    req.resolve("answered", ["English"])
    req.resolve("cancelled", ["ignored"])
    assert req.state == "answered" and req.answers == ["English"]


def test_a_never_resolved_request_does_not_deadlock_the_worker():
    # The worker's wait() is bounded, so a dead main thread cannot hang it.
    req = ClarifyRequest(QS)
    done = threading.Event()

    def worker():
        if not req.done.wait(0.2):
            req.resolve("timeout")
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(2.0)
    assert done.is_set() and req.state == "timeout"


def test_deadline_expires_and_can_be_extended():
    clock = {"t": 0.0}
    req = ClarifyRequest(QS, timeout=10.0, clock=lambda: clock["t"])
    assert not req.expired()
    clock["t"] = 11.0
    assert req.expired()
    req.extend_deadline()
    assert not req.expired()


def test_timeout_is_longer_than_the_key_capture_deadline():
    # Reading options and choosing takes longer than pressing one key (10 s).
    assert CLARIFY_TIMEOUT > 10.0


# ------------------------------------------------------------- plan parsing

def test_parse_clarify_reads_a_well_formed_plan():
    assert parse_clarify(
        '{"questions": [{"question": "用咩語言寫？",'
        ' "options": ["English", "繁體中文"]}]}') == QS


def test_parse_clarify_treats_anything_malformed_as_no_questions():
    for raw in ("garbage", "", "[]", "null", "{}", '{"questions": "nope"}',
                '{"questions": [{"question": "q"}]}',
                '{"questions": [{"options": ["a", "b"]}]}',
                '{"questions": [42]}'):
        assert parse_clarify(raw) == [], raw


def test_parse_clarify_drops_a_question_with_fewer_than_two_options():
    assert parse_clarify(
        '{"questions": [{"question": "q", "options": ["only"]}]}') == []


def test_parse_clarify_clamps_questions_and_options():
    raw = ('{"questions": [' + ",".join(
        '{"question": "q%d", "options": ["a","b","c","d","e"]}' % i
        for i in range(5)) + "]}")
    out = parse_clarify(raw)
    assert len(out) == MAX_CLARIFY_QUESTIONS
    assert all(len(q["options"]) == MAX_CLARIFY_OPTIONS for q in out)


def test_parse_clarify_drops_blank_options_and_questions():
    out = parse_clarify(
        '{"questions": [{"question": "  ", "options": ["a", "b"]},'
        ' {"question": "q", "options": ["a", "  ", "b"]}]}')
    assert out == [{"question": "q", "options": ["a", "b"]}]
