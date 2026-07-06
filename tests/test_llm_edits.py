"""Edit-list correction layer: the LLM proposes {"from","to"} edits and the
code applies only the ones that pass deterministic safety guards — a
hallucinated edit must become a no-op, never corrupted output."""
from whisperflow_local.llm import parse_edits
from whisperflow_local.textproc import apply_edits


# ------------------------------------------------------------ parse_edits

def test_parse_edits_happy_path():
    raw = '{"edits": [{"from": "中影夾雜", "to": "中英夾雜"}]}'
    assert parse_edits(raw) == [{"from": "中影夾雜", "to": "中英夾雜"}]


def test_parse_edits_garbage_returns_empty():
    assert parse_edits("not json at all") == []
    assert parse_edits('{"edits": "oops"}') == []
    assert parse_edits('{"other": []}') == []
    assert parse_edits('{"edits": [{"from": "x"}, "junk", {"to": "y"}]}') == []


# ------------------------------------------------------------ apply_edits

def test_apply_edits_fixes_homophone():
    out = apply_edits("我中影夾雜噉講嘢", [{"from": "中影", "to": "中英"}])
    assert out == "我中英夾雜噉講嘢"


def test_apply_edits_ignores_edit_not_in_text():
    # The model hallucinating a fragment that does not exist must no-op.
    out = apply_edits("幫我 send 個 email", [{"from": "即係", "to": ""}])
    assert out == "幫我 send 個 email"


def test_apply_edits_rejects_cross_script_translation():
    # send→發 is a translation, not a homophone fix. Blocked by guard.
    out = apply_edits("幫我 send 個 email", [{"from": "send", "to": "發"}])
    assert out == "幫我 send 個 email"


def test_apply_edits_rejects_single_cjk_char():
    # 影 appears three times; single-char replace-all is too blunt to trust.
    text = "會唔會影得好啲？中影夾雜會影得更好？"
    assert apply_edits(text, [{"from": "影", "to": "聽"}]) == text


def test_apply_edits_rejects_variant_char_churn():
    # 一個→一箇 is the same word in a variant spelling (t2s-identical).
    text = "幫我整一個新嘅 feature"
    assert apply_edits(text, [{"from": "一個", "to": "一箇"}]) == text


def test_apply_edits_rejects_long_rewrites():
    text = "請幫我編譯出一個新嘅 feature 出嚟啦"
    edit = [{"from": "請幫我編譯出一個新嘅 feature", "to": "請幫我寫一個新功能"}]
    assert apply_edits(text, edit) == text


def test_apply_edits_deletes_fillers_only():
    assert apply_edits("呃我想睇下天氣", [{"from": "呃", "to": ""}]) == "我想睇下天氣"
    # Deleting arbitrary words via empty "to" is blocked.
    assert apply_edits("我想睇下天氣", [{"from": "想", "to": ""}]) == "我想睇下天氣"


def test_apply_edits_collapses_leftover_spaces():
    out = apply_edits("um let me think", [{"from": "um", "to": ""}])
    assert out == "let me think"


# ------------------------------------------------- phonetic guard (jyutping)

def test_homophone_edit_passes_phonetic_guard():
    # 影 jing2 / 認 jing6 and 影 jing2 / 英 jing1 differ only in tone.
    text = "會唔會影得好啲？我中影夾雜。"
    out = apply_edits(text, [{"from": "影得", "to": "認得"},
                             {"from": "中影", "to": "中英"}])
    assert out == "會唔會認得好啲？我中英夾雜。"


def test_wrong_guess_fails_phonetic_guard():
    # 錄 luk6 sounds nothing like 影 jing2 — a wrong-guess replacement
    # observed live from qwen3.5:4b. Must be rejected.
    text = "會唔會影得好少少？"
    assert apply_edits(text, [{"from": "影得", "to": "錄得"}]) == text


def test_latin_vocab_recovery_allows_longer_target():
    out = apply_edits("一隻字咁樣去入 Word 度",
                      [{"from": "Word", "to": "hot word"}])
    assert out == "一隻字咁樣去入 hot word 度"
