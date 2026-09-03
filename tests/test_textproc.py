"""Unit tests for tag stripping, dictionary and voice commands."""
from whisperflow_local.textproc import (
    apply_dictionary,
    parse_voice_commands,
    strip_sensevoice_tags,
)

# ------------------------------------------------------------ tag stripping (B1)

def test_strip_language_emotion_event_tags():
    raw = "<|yue|><|NEUTRAL|><|Speech|><|withitn|>你今日食咗飯未呀 I am busy"
    assert strip_sensevoice_tags(raw) == "你今日食咗飯未呀 I am busy"


def test_strip_tags_between_segments():
    raw = "<|en|><|HAPPY|><|Speech|>hello there <|yue|><|NEUTRAL|><|Speech|>係呀"
    assert strip_sensevoice_tags(raw) == "hello there 係呀"


def test_strip_preserves_plain_text():
    assert strip_sensevoice_tags("no tags here 冇標籤") == "no tags here 冇標籤"


# ------------------------------------------------------------ dictionary (D1)

def test_dictionary_latin_case_insensitive_word_boundary():
    entries = [{"from": "kenny", "to": "Ken Ng"}]
    assert apply_dictionary("tell Kenny about it", entries) == "tell Ken Ng about it"
    # No partial-word replacement:
    assert apply_dictionary("kennyville", entries) == "kennyville"


def test_dictionary_cjk_exact():
    entries = [{"from": "維史巴", "to": "WhisperFlow"}]
    assert apply_dictionary("用維史巴寫嘢", entries) == "用WhisperFlow寫嘢"


def test_dictionary_empty_entries_no_change():
    assert apply_dictionary("unchanged", []) == "unchanged"
    assert apply_dictionary("unchanged", [{"from": "", "to": "x"}]) == "unchanged"


# ------------------------------------------------------------ voice commands (C3)

def test_new_line_inline():
    p = parse_voice_commands("first item new line second item")
    assert p.text == "first item\nsecond item"
    assert not p.press_enter


def test_new_paragraph_cantonese():
    p = parse_voice_commands("第一段 新段落 第二段")
    assert p.text == "第一段\n\n第二段"


def test_trailing_send_sets_enter():
    p = parse_voice_commands("see you at six press enter")
    assert p.text == "see you at six"
    assert p.press_enter


def test_trailing_send_cantonese():
    p = parse_voice_commands("好呀聽日見 發送")
    assert p.text == "好呀聽日見"
    assert p.press_enter


def test_scratch_that_alone():
    p = parse_voice_commands("scratch that")
    assert p.scratch and p.text == ""


def test_scratch_cantonese():
    p = parse_voice_commands("當我冇講過")
    assert p.scratch


def test_scratch_not_triggered_mid_sentence():
    p = parse_voice_commands("do not scratch that surface with a knife")
    assert not p.scratch
    assert "scratch that" in p.text


def test_all_caps_prefix():
    p = parse_voice_commands("all caps urgent call me")
    assert p.text == "URGENT CALL ME"


def test_plain_text_untouched():
    p = parse_voice_commands("今日開會 we should sync about the roadmap")
    assert p.text == "今日開會 we should sync about the roadmap"
    assert not p.press_enter and not p.scratch
