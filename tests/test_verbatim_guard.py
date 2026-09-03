"""The Verbatim rewrite guard: what the LLM is allowed to change.

The guard is what makes a whole-utterance rewrite safe enough to use for
punctuation repair (an {"from","to"} edit list cannot move a 。). It accepts
only deletions and punctuation/case changes; everything else degrades to the
original text.
"""
from whisperflow_local.llm import parse_cleanup
from whisperflow_local.textproc import (
    SKELETON_KEEP_MIN,
    guard_verbatim,
    is_subsequence,
    skeleton,
)

BASE = ("今日系五月二十六號，咁我先記錄下，誒，今日有啲乜嘢代辦事項啦。"
        "咁個 to do list 系啲乜嘢咧？")


# ------------------------------------------------------------- primitives

def test_skeleton_drops_punctuation_whitespace_and_case():
    assert skeleton("你好，世界！") == "你好世界"
    assert skeleton("Hello, World.") == "helloworld"
    assert skeleton("幫我 book 三點") == "幫我book三點"


def test_is_subsequence():
    assert is_subsequence("abc", "axbxc")
    assert is_subsequence("", "anything")
    assert not is_subsequence("acb", "abc")   # order matters
    assert not is_subsequence("abcd", "abc")  # insertion


# ------------------------------------------------------------- accepted

def test_removing_fillers_and_fixing_punctuation_is_accepted():
    cleaned = ("今日系五月二十六號，咁我先記錄下，今日有啲乜嘢代辦事項啦。"
               "咁個 to do list 系啲乜嘢咧？")
    assert guard_verbatim(BASE, cleaned) == cleaned


def test_spurious_question_mark_can_be_rewritten_to_a_full_stop():
    base = "我先記錄下今日有啲乜嘢代辦事項啦？"
    cleaned = "我先記錄下今日有啲乜嘢代辦事項啦。"
    assert guard_verbatim(base, cleaned) == cleaned


def test_english_capitalisation_is_accepted():
    base = "睇下 codex 有冇更新"
    assert guard_verbatim(base, "睇下 CodeX 有冇更新") == "睇下 CodeX 有冇更新"


def test_stutter_removal_is_accepted():
    base = "咁我，我先記錄下"
    assert guard_verbatim(base, "咁我先記錄下") == "咁我先記錄下"


# ------------------------------------------------------------- rejected

def test_written_chinese_conversion_is_rejected():
    cleaned = "今天是5月26號，我記錄一下今天的代辦事項。"
    assert guard_verbatim(BASE, cleaned) == BASE


def test_translation_is_rejected():
    assert guard_verbatim(BASE, "Today is May 26. Let me record my to-do list.") == BASE


def test_summarising_is_rejected_by_the_length_floor():
    # A summary is trivially a subsequence, so only the floor catches it.
    assert is_subsequence(skeleton("今日代辦事項"), skeleton(BASE))
    assert guard_verbatim(BASE, "今日代辦事項") == BASE


def test_word_substitution_is_rejected():
    base = "please send the report"
    assert guard_verbatim(base, "please sent the report") == base


def test_reordering_is_rejected():
    base = "我食飯之後去街"
    assert guard_verbatim(base, "我去街之後食飯") == base


def test_inserted_words_are_rejected():
    base = "我今日好忙"
    assert guard_verbatim(base, "我今日真係好忙") == base


def test_empty_or_blank_rewrite_falls_back():
    assert guard_verbatim(BASE, "") == BASE
    assert guard_verbatim(BASE, "   ") == BASE


def test_keep_min_is_configurable_and_default_is_sane():
    assert 0.0 < SKELETON_KEEP_MIN < 1.0
    base = "一二三四五六七八九十"
    assert guard_verbatim(base, "一二三四五", keep_min=0.4) == "一二三四五"
    assert guard_verbatim(base, "一二三四五", keep_min=0.9) == base


# ------------------------------------------------------- cleanup parsing

def test_parse_cleanup_reads_both_channels():
    out = parse_cleanup('{"clean": "你好。", "edits": [{"from": "ab", "to": "cd"}]}')
    assert out == {"clean": "你好。", "edits": [{"from": "ab", "to": "cd"}]}


def test_parse_cleanup_degrades_to_a_no_op():
    # Every malformed shape must yield an empty clean (guard keeps the
    # original) and no edits — never an exception.
    for raw in ("not json", "", "[1, 2]", "null",
                '{"edits": []}', '{"clean": 42, "edits": "nope"}'):
        assert parse_cleanup(raw) == {"clean": "", "edits": []}, raw


def test_a_rejected_rewrite_still_lets_the_edits_through():
    # The two channels are independent: a hallucinated rewrite must not cost
    # the user a legitimate homophone fix.
    from whisperflow_local.textproc import apply_edits
    base = "我想食意大利份"
    guarded = guard_verbatim(base, "I want Italian pasta")   # rejected
    assert guarded == base
    assert apply_edits(guarded, [{"from": "食意大利份", "to": "食意大利粉"}]) == "我想食意大利粉"
