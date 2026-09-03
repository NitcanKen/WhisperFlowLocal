"""Clarify panel: geometry, key mapping, and the worker/main-thread handoff.

Only the pure half is tested — no NSPanel is created, the same way
tests/test_insert_and_overlay.py covers overlay.py's module-level helpers.
"""
import threading

from whisperflow_local.clarify import (
    CLARIFY_TIMEOUT,
    CLARIFY_W,
    CLOSE_GATE_FLOOR,
    CONTENT_GATE,
    CONTENT_RISE,
    PAD,
    ClarifyRequest,
    blend_rect,
    close_gate,
    content_alpha,
    content_offset,
    field_text_rect,
    morph_rect,
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


# ------------------------------------------------------------- motion
# The card is born at the pill's rect and grows out of it, so the two read as
# one object rather than two windows appearing in sequence.

PILL = (100.0, 96.0, 300.0, 44.0)
CARD = (60.0, 96.0, 380.0, 202.0)


def test_morph_starts_at_the_pill_and_ends_at_the_card():
    assert morph_rect(0.0, True, PILL, CARD) == PILL
    assert morph_rect(1.0, True, PILL, CARD) == CARD


def test_morph_keeps_the_bottom_edge_pinned():
    # Both rects share a bottom edge, so every frame must too — that is what
    # makes it read as the pill growing upward.
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert morph_rect(p, True, PILL, CARD)[1] == PILL[1]


def test_morph_overshoots_while_opening_but_not_while_closing():
    # Opening borrows the pill's ease_out_back, so width may briefly exceed
    # the card's; closing is ease_in_cubic and must never overshoot.
    assert max(morph_rect(p / 20.0, True, PILL, CARD)[2]
               for p in range(21)) > CARD[2]
    assert max(morph_rect(p / 20.0, False, PILL, CARD)[2]
               for p in range(21)) <= CARD[2] + 1e-9


def test_blend_rect_absorbs_a_target_change_without_jumping():
    # Question 2 may have a different option count; the height must ease.
    cur = blend_rect(None, PILL)
    assert cur == PILL
    nxt = blend_rect(cur, CARD)
    assert PILL[3] < nxt[3] < CARD[3]


def test_content_reveal_is_staggered_and_ordered():
    alphas = [content_alpha(0.5, i, 5) for i in range(5)]
    assert alphas == sorted(alphas, reverse=True)   # earlier elements lead
    assert alphas[0] == 1.0 and alphas[-1] < 1.0


def test_every_element_is_fully_revealed_by_the_end():
    for count in (3, 4, 5, 6):
        assert all(content_alpha(1.0, i, count) == 1.0 for i in range(count))


def test_nothing_is_revealed_at_the_start():
    assert all(content_alpha(0.0, i, 5) == 0.0 for i in range(1, 5))


def test_text_prints_while_the_box_is_still_growing():
    # Waiting for the box to finish leaves a beat where a full-size empty
    # card just sits there, which is what made it feel bolted on.
    assert CONTENT_GATE < 1.0
    assert close_gate(0.9, True) > 0.0


def test_text_clears_before_the_box_folds_shut():
    assert close_gate(CLOSE_GATE_FLOOR, False) == 0.0
    assert close_gate(0.3, False) == 0.0        # box still 30% open, text gone


def test_content_offset_rises_into_place():
    assert content_offset(0.0) == CONTENT_RISE
    assert content_offset(1.0) == 0.0


# ------------------------------------------------------------- esc / skip

def test_field_text_line_is_centred_in_its_ground():
    # An NSTextField lays text out at the TOP of its frame, so a field sized
    # to the whole ground types in the top-left corner.
    ground = other_rect(2)
    fx, fy, fw, fh = field_text_rect(ground)
    gx, gy, gw, gh = ground
    assert fh < gh
    assert abs((fy - gy) - ((gy + gh) - (fy + fh))) < 1e-9   # equal margins
    assert fx > gx and fx + fw < gx + gw                     # inset both sides


def test_skipped_still_carries_the_answers_given_so_far():
    req = ClarifyRequest([{"question": "q1", "options": ["a", "b"]},
                          {"question": "q2", "options": ["c", "d"]}])
    req.answers.append("a")
    req.resolve("skipped", req.answers)
    assert req.state == "skipped" and req.answers == ["a"]


def test_generate_pairs_only_the_answered_questions():
    # Esc leaves later questions unanswered; they must simply not appear in
    # the prompt rather than desyncing the pairing.
    from whisperflow_local.llm import BaseLLMBackend
    seen = {}

    class B(BaseLLMBackend):
        def _chat(self, system, user, force_json=False):
            seen.update(system=system, user=user)
            return "written"

    qs = [{"question": "用咩語言寫？", "options": ["English", "繁體中文"]},
          {"question": "幾正式？", "options": ["正式", "輕鬆"]}]
    B("http://x", "m").generate("draft an email", questions=qs, answers=["English"])
    assert "用咩語言寫？" in seen["system"] and "English" in seen["system"]
    assert "幾正式？" not in seen["system"]    # unanswered: must not appear
    assert "English" in seen["user"]


# ---------------------------------------------------- clarify answers bind
# Reported: typing "simplified Chinese" into the free-text field still
# produced traditional. The answer WAS captured (the log showed
# "clarify answered: ['Simplified Chinese']") — it just never bound.

def test_answers_are_restated_at_the_end_of_the_user_message():
    # Measured against the live model: in the system prompt alone the answer
    # loses to the request's own language. Repeating it last is what binds it.
    from whisperflow_local.llm import BaseLLMBackend
    seen = {}

    class B(BaseLLMBackend):
        def _chat(self, system, user, force_json=False):
            seen.update(system=system, user=user)
            return "written"

    qs = [{"question": "用咩語言寫？", "options": ["English", "繁體中文"]}]
    B("http://x", "m").generate("幫我寫封 email", questions=qs,
                                answers=["简体中文"])
    assert "简体中文" in seen["system"]          # as a hard constraint...
    assert "简体中文" in seen["user"]            # ...and restated last
    assert seen["user"].rstrip().endswith("）")


def test_no_trailing_restatement_when_nothing_was_answered():
    from whisperflow_local.llm import BaseLLMBackend
    seen = {}

    class B(BaseLLMBackend):
        def _chat(self, system, user, force_json=False):
            seen["user"] = user
            return "written"

    B("http://x", "m").generate("幫我寫封 email", questions=[], answers=[])
    assert seen["user"] == "幫我寫封 email"


def test_requested_script_reads_either_phrasing():
    from whisperflow_local.textproc import requested_script
    for answer in ("Simplified Chinese", "简体中文", "簡體", "zh-Hans", "hans"):
        assert requested_script([answer]) == "simplified", answer
    for answer in ("繁體中文", "Traditional Chinese", "zh-HK", "hant"):
        assert requested_script([answer]) == "traditional", answer
    for answer in ("English", "日本語", "正式", "", "  "):
        assert requested_script([answer]) is None, answer
    assert requested_script([]) is None
    assert requested_script(["正式", "简体中文"]) == "simplified"


def test_script_conversion_is_deterministic_and_leaves_latin_alone():
    from whisperflow_local.textproc import to_hk, to_simplified
    assert to_simplified("我們已完成開發工作") == "我们已完成开发工作"
    assert to_hk("我们已完成开发工作") == "我們已完成開發工作"
    assert to_simplified("We have completed it.") == "We have completed it."
