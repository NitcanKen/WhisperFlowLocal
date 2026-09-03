"""Tests for zh-HK localization and the punctuation toggle."""
from whisperflow_local import i18n
from whisperflow_local.i18n import S, set_language, tr
from whisperflow_local.textproc import strip_punctuation


def teardown_function(_):
    set_language("en")


# ------------------------------------------------------------- i18n

def test_english_and_zh_hk_strings():
    set_language("en")
    assert tr("menu_settings") == "Settings"
    set_language("zh-HK")
    assert tr("menu_settings") == "設定"
    assert tr("menu_quit") == "結束"
    assert tr("menu_profile") == "格式化模式"


def test_formatting_kwargs():
    set_language("zh-HK")
    out = tr("status_llm_off_raw", err="connection refused")
    assert "connection refused" in out and "原始轉寫" in out


def test_every_key_has_both_translations():
    for key, pair in S.items():
        assert isinstance(pair, tuple) and len(pair) == 2, key
        assert pair[0] and pair[1], key


def test_unknown_key_returns_key():
    assert tr("no_such_key_xyz") == "no_such_key_xyz"


def test_auto_follows_system():
    set_language("auto")
    assert i18n.is_chinese() == i18n.system_prefers_chinese()


# ------------------------------------------------------------- punctuation

def test_strip_punctuation_cjk_and_latin():
    assert strip_punctuation("你好，世界！Hello, world.") == "你好 世界 Hello world"


def test_strip_punctuation_keeps_newlines():
    out = strip_punctuation("第一行，好。\n第二行！")
    assert out == "第一行 好\n第二行"


def test_strip_punctuation_plain_text_unchanged():
    assert strip_punctuation("冇標點 no punct here") == "冇標點 no punct here"
