"""Pure text processing: SenseVoice tag stripping, custom dictionary,
voice-command parsing, and per-app profile selection.

Everything here is deterministic and covered by unit tests.
"""
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- tag stripping

_TAG_RE = re.compile(r"<\|[^|<>]*\|>")
# rich_transcription_postprocess in funasr may map tags to emoji; strip those too.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F02F"
    "\U0001F900-\U0001F9FF\U00002600-\U000026FF]"
)


def strip_sensevoice_tags(text: str) -> str:
    """Remove SenseVoice metadata tags like <|yue|>, <|NEUTRAL|>, <|Speech|>."""
    text = _TAG_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    # Collapse runs of whitespace introduced by tag removal.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ------------------------------------------------------------- punctuation

_PUNCT_CHARS = (
    "，。、！？；：…‧·「」『』（）《》〈〉【】,.!?;:\"'()\\[\\]{}<>~—–-"
)
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")


def strip_punctuation(text: str) -> str:
    """Remove punctuation (user toggle), keeping newlines from voice
    commands and collapsing the leftover double spaces."""
    out = _PUNCT_RE.sub(" ", text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return out.strip()


# ---------------------------------------------------------------- dictionary

def apply_dictionary(text: str, entries: list) -> str:
    """Apply user word replacements.

    Latin-script sources match case-insensitively on word boundaries;
    CJK sources match exactly (no word boundaries in CJK).
    """
    for entry in entries or []:
        src = str(entry.get("from", ""))
        dst = str(entry.get("to", ""))
        if not src:
            continue
        if re.search(r"[a-zA-Z]", src) and not re.search(r"[一-鿿]", src):
            pattern = re.compile(
                r"(?<![a-zA-Z0-9])" + re.escape(src) + r"(?![a-zA-Z0-9])",
                re.IGNORECASE,
            )
            text = pattern.sub(dst, text)
        else:
            text = text.replace(src, dst)
    return text


# ------------------------------------------------------- fast-path cleanup
# Clean speech skips the LLM entirely (the ~1.3 s latency budget cannot fit
# a 4B decode): deterministic script conversion + spacing + vocabulary
# casing. Utterances with disfluencies still route to the LLM.

_LATIN_FILLER_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s,，。.!?！？]))(?:um+|uh+|erm?|hmm+)"
    r"(?=$|[\s,，。.!?！？])",
    re.IGNORECASE,
)
_CJK_FILLER_RE = re.compile(r"[呃嗯]|即係")
_LATIN_STUTTER_RE = re.compile(r"\b([a-zA-Z]+)\s+\1\b", re.IGNORECASE)

_CJK = r"一-鿿"
_CC_S2HK = None


def needs_llm_cleanup(text: str) -> bool:
    """True when the utterance has disfluencies only an LLM can judge
    (fillers, '即係' which may or may not be filler, Latin stutters).
    CJK reduplication ('試試', '謝謝') is grammatical, so it never triggers."""
    return bool(
        _LATIN_FILLER_RE.search(text)
        or _CJK_FILLER_RE.search(text)
        or _LATIN_STUTTER_RE.search(text)
    )


def to_hk(text: str) -> str:
    """Simplified -> Hong Kong traditional (OpenCC s2hk). ASR models emit
    simplified even for Cantonese; HK users expect 繁體. No-op if OpenCC
    is unavailable or the text is already traditional/Latin."""
    global _CC_S2HK
    if _CC_S2HK is None:
        try:
            from opencc import OpenCC
            _CC_S2HK = OpenCC("s2hk")
        except Exception:
            _CC_S2HK = False
    return _CC_S2HK.convert(text) if _CC_S2HK else text


def quick_clean(text: str, vocab: list = None, hk: bool = True) -> str:
    """Deterministic cleanup for the no-LLM fast path: HK script, CJK/Latin
    spacing normalization, punctuation spacing, canonical vocabulary casing."""
    if hk:
        text = to_hk(text)
    # One space at CJK<->Latin boundaries, none between CJK characters.
    text = re.sub(rf"(?<=[{_CJK}])(?=[A-Za-z0-9])", " ", text)
    text = re.sub(rf"(?<=[A-Za-z0-9])(?=[{_CJK}])", " ", text)
    text = re.sub(rf"(?<=[{_CJK}]) +(?=[{_CJK}])", "", text)
    # No space before punctuation (either script).
    text = re.sub(r" +(?=[，。！？、；：,.!?;:])", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Canonical casing/spelling for Latin vocabulary terms.
    for term in vocab or []:
        if re.search(r"[a-zA-Z]", term):
            pattern = re.compile(
                r"(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])",
                re.IGNORECASE,
            )
            text = pattern.sub(term, text)
    return text.strip()


# ---------------------------------------------------------------- vocabulary

def parse_vocab_entry(raw: str):
    """Parse the vocabulary-add input.

    '錯字 → 正字' / 'wrong -> right'  →  ("pair", src, dst)
    'WhisperFlow'                     →  ("term", term)
    Malformed/empty input             →  None
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    for arrow in ("→", "->"):
        if arrow in raw:
            src, _, dst = raw.partition(arrow)
            src, dst = src.strip(), dst.strip()
            if src and dst:
                return ("pair", src, dst)
            return None
    return ("term", raw)


def vocab_terms(dictionary: list, hotwords: list) -> list:
    """The unified bias list: bare hotwords plus every replacement target,
    deduped, order preserved. Fed to the LLM prompt (and, for engines that
    support it, ASR context biasing)."""
    seen, out = set(), []
    for term in list(hotwords or []) + [
        str(e.get("to", "")) for e in (dictionary or [])
    ]:
        t = str(term).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------- voice commands

@dataclass
class ParsedUtterance:
    text: str
    press_enter: bool = False
    scratch: bool = False  # discard the previous dictation
    actions: list = field(default_factory=list)


_PUNCT_EDGE = r"[\s,.。，、!！?？]*"

# Phrases that map to literal insertions when spoken on their own segment.
_NEWLINE_PHRASES = ["new line", "newline", "換行", "换行"]
_PARAGRAPH_PHRASES = ["new paragraph", "next paragraph", "新段落", "另起一段"]
_SCRATCH_PHRASES = [
    "scratch that", "delete that", "刪除啱先嗰句", "删除刚才那句", "當我冇講過",
]
_SEND_PHRASES = ["press enter", "send it", "send message", "發送", "发送"]
_ALLCAPS_PREFIX = ["all caps", "全部大寫", "全部大写"]


def _phrase_re(phrases):
    alt = "|".join(re.escape(p) for p in phrases)
    return re.compile(
        _PUNCT_EDGE + r"(?<![a-zA-Z])(" + alt + r")(?![a-zA-Z])" + _PUNCT_EDGE + r"$",
        re.IGNORECASE,
    )


def _inline_re(phrases):
    alt = "|".join(re.escape(p) for p in phrases)
    return re.compile(
        r"[\s,，、]*(?<![a-zA-Z])(" + alt + r")(?![a-zA-Z])[\s,，、]*",
        re.IGNORECASE,
    )


_SEND_RE = _phrase_re(_SEND_PHRASES)
_NEWLINE_INLINE = _inline_re(_NEWLINE_PHRASES)
_PARAGRAPH_INLINE = _inline_re(_PARAGRAPH_PHRASES)


def parse_voice_commands(text: str) -> ParsedUtterance:
    """Detect spoken commands and apply them instead of typing them literally.

    Rules:
      - "new line" / "new paragraph" (or Cantonese equivalents) anywhere in the
        utterance become \n / \n\n.
      - A trailing "press enter" / "send it" / 發送 sets press_enter.
      - An utterance that is ONLY "scratch that" (or 刪除啱先嗰句 / 當我冇講過)
        requests deletion of the previous dictation.
      - A leading "all caps" uppercases the remainder of the utterance.
    """
    result = ParsedUtterance(text=text.strip())
    if not result.text:
        return result

    stripped = result.text.strip().strip(".。,，!！?？ ").lower()
    for phrase in _SCRATCH_PHRASES:
        if stripped == phrase.lower():
            result.text = ""
            result.scratch = True
            result.actions.append("scratch")
            return result

    m = _SEND_RE.search(result.text)
    if m:
        result.text = result.text[: m.start()].rstrip()
        result.press_enter = True
        result.actions.append("enter")

    for prefix in _ALLCAPS_PREFIX:
        pat = re.compile(r"^" + re.escape(prefix) + r"\b[\s,，:：]*", re.IGNORECASE)
        m = pat.match(result.text)
        if m:
            result.text = result.text[m.end():].upper()
            result.actions.append("allcaps")
            break

    result.text = _PARAGRAPH_INLINE.sub("\n\n", result.text)
    result.text = _NEWLINE_INLINE.sub("\n", result.text)
    result.text = result.text.strip()
    return result


# ---------------------------------------------------------------- profile rules

def pick_profile(app_name: str, rules: dict, default_profile: str) -> str:
    """Choose a formatting profile from the frontmost app name.

    Rule keys match as case-insensitive substrings of the app name.
    Longest key wins so 'iTerm' beats 'Term'-style overlaps deterministically.
    """
    if not app_name:
        return default_profile
    lowered = app_name.lower()
    best_key = None
    for key in (rules or {}):
        if key.lower() in lowered:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is not None:
        return rules[best_key]
    return default_profile
