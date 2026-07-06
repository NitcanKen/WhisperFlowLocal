"""Deterministic jyutping phonetic hot-word recovery: SenseVoice has no
model-level biasing, so a known hot-word the ASR mis-transcribed as a
homophone (中影夾雜 for 中英夾雜, 適別 for 識別) is recovered here with zero
network and ~0 latency. Anything not phonetically near a hot-word is left
exactly as-is — the layer must never corrupt correct text."""
from whisperflow_local.textproc import apply_phonetic_hotwords


# ------------------------------------------------------- CJK homophone recovery

def test_recovers_cjk_homophone_multichar_hotword():
    # 中影夾雜 and 中英夾雜 are toneless-jyutping identical (影 jing2 / 英 jing1).
    out = apply_phonetic_hotwords("我中影夾雜噉講嘢", ["中英夾雜"])
    assert out == "我中英夾雜噉講嘢"


def test_recovers_cjk_homophone_two_char_hotword():
    # 適別 → 識別 (both sik-bit).
    out = apply_phonetic_hotwords("轉咗做適別語言", ["識別"])
    assert out == "轉咗做識別語言"


def test_correct_hotword_occurrence_is_left_unchanged():
    assert apply_phonetic_hotwords("轉咗做識別語言", ["識別"]) == "轉咗做識別語言"


# ------------------------------------------------------------ no false positives

def test_no_op_when_nothing_phonetically_near():
    assert apply_phonetic_hotwords("今晚一齊食飯", ["識別"]) == "今晚一齊食飯"


def test_phonetically_distant_pair_is_rejected():
    # 影得 (jing-dak) is nowhere near 錄音 (luk-jam): must NOT be replaced.
    assert apply_phonetic_hotwords("會唔會影得好少少", ["錄音"]) == "會唔會影得好少少"


def test_unrelated_correct_word_not_pulled_to_hotword():
    # 實別 (sat-bit) is > 1 edit from 識別 (sik-bit) — stays put.
    assert apply_phonetic_hotwords("呢個實別嘅嘢", ["識別"]) == "呢個實別嘅嘢"


# ------------------------------------------------------------ Latin fuzzy casing

def test_latin_single_typo_recovered_to_canonical():
    out = apply_phonetic_hotwords("用緊 whisperflw 好正", ["WhisperFlow"])
    assert out == "用緊 WhisperFlow 好正"


def test_latin_correct_token_unchanged():
    assert apply_phonetic_hotwords("用緊 WhisperFlow", ["WhisperFlow"]) == "用緊 WhisperFlow"


def test_latin_unrelated_word_not_matched():
    assert apply_phonetic_hotwords("I read a book today", ["WhisperFlow"]) \
        == "I read a book today"


def test_latin_short_hotword_not_fuzzy_matched():
    # Short terms (< 6 letters) are too collision-prone for distance-1 fuzz.
    assert apply_phonetic_hotwords("I book a room", ["Look"]) == "I book a room"


# ------------------------------------------------------------ edges

def test_empty_text():
    assert apply_phonetic_hotwords("", ["識別"]) == ""


def test_no_hotwords_is_identity():
    assert apply_phonetic_hotwords("我中影夾雜", []) == "我中影夾雜"
    assert apply_phonetic_hotwords("我中影夾雜", None) == "我中影夾雜"
