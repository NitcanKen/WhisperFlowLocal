"""Fast-path routing: disfluency detection and deterministic cleanup."""
from whisperflow_local.textproc import needs_llm_cleanup, quick_clean, to_hk


# ------------------------------------------------------------ routing

def test_clean_speech_skips_llm():
    assert not needs_llm_cleanup("幫我 send 個 email 俾 David")
    assert not needs_llm_cleanup("book a table for two at eight")


def test_fillers_route_to_llm():
    assert needs_llm_cleanup("um let me think about it")
    assert needs_llm_cleanup("so uh we should go")
    assert needs_llm_cleanup("呃我想講嘅係")
    assert needs_llm_cleanup("嗯好啊")
    assert needs_llm_cleanup("即係我想話俾你知")


def test_latin_stutter_routes_to_llm():
    assert needs_llm_cleanup("send the the email")


def test_cjk_reduplication_is_not_a_stutter():
    # 試試/謝謝-style reduplication is grammatical Cantonese/Chinese.
    assert not needs_llm_cleanup("等我試試先")
    assert not needs_llm_cleanup("多謝謝謝")


def test_filler_words_inside_english_words_do_not_trigger():
    assert not needs_llm_cleanup("the drummer hums a tune")  # um/hmm embedded


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
