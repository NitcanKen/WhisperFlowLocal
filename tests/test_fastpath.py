"""LLM routing and the deterministic cleanup used when the LLM is off/down."""
from whisperflow_local.textproc import quick_clean, should_use_llm, to_hk


# ------------------------------------------------------------ routing

def test_fluent_speech_still_routes_to_llm():
    # Regression: ASR homophone slips (影得→認得, 中影→中英) carry no
    # disfluency marker, so routing on disfluencies starved the LLM of
    # exactly the text it exists to fix.
    assert should_use_llm(True, "幫我 send 個 email 俾 David")
    assert should_use_llm(True, "book a table for two at eight")


def test_every_non_empty_utterance_uses_llm():
    # Raw is gone: both surviving profiles need the model, so routing no
    # longer depends on the profile at all.
    assert should_use_llm(True, "今晚八點食飯")


def test_llm_disabled_skips_llm():
    assert not should_use_llm(False, "幫我 send 個 email")


def test_empty_text_skips_llm():
    assert not should_use_llm(True, "   ")


# ------------------------------------------------------------ quick_clean

def test_quick_clean_converts_to_hk_traditional():
    assert to_hk("讲今晚八点食饭") == "講今晚八點食飯"
    assert quick_clean("讲今晚八点食饭") == "講今晚八點食飯"


def test_quick_clean_normalizes_cjk_latin_spacing():
    out = quick_clean("帮我 send个 email 俾 david 。")
    assert out == "幫我 send 個 email 俾 david。"


def test_quick_clean_applies_vocab_casing():
    # note: s2hk also normalizes 搵 to the HK-standard variant 揾
    out = quick_clean("幫我搵 david 去 central", vocab=["David", "Central"])
    assert out == "幫我揾 David 去 Central"


def test_quick_clean_keeps_latin_only_text():
    assert quick_clean("send the email to David") == "send the email to David"


def test_quick_clean_no_hk_flag():
    assert quick_clean("讲今晚", hk=False) == "讲今晚"
