#!/usr/bin/env python
"""Live end-to-end check for the Clean post-correction pipeline.

Part A: real SenseVoice transcription of the last recording — proves the ASR
engine still loads and runs after the post-correction changes.

Part B: the EXACT shipped Clean pipeline
  quick_clean -> apply_phonetic_hotwords -> llm.propose_edits -> apply_edits
run against the real SenseVoice outputs captured in app.log (per-sentence
audio is not retained, so we feed the genuine ASR strings), with real
Qwen3.6-35B-A3B via GB10 vLLM. Prints before -> after so corrections and the
must-not-change regressions are both visible.

Usage: .venv/bin/python scripts/itest_postcorrect_live.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whisperflow_local import paths  # noqa: E402
from whisperflow_local.asr import ASREngine  # noqa: E402
from whisperflow_local.config import Config  # noqa: E402
from whisperflow_local.llm import VLLMBackend  # noqa: E402
from whisperflow_local.textproc import (  # noqa: E402
    apply_edits,
    apply_phonetic_hotwords,
    quick_clean,
    vocab_terms,
)

# (real SenseVoice ASR output from app.log, note) — the target fix and the
# regressions that must survive untouched.
FIELD = [
    ("咁我而家轉咗做識別語言為自動，咁睇下佢會出啲咩語言俾我，同埋會唔會影得好少少？"
     "定係我中影夾雜嘅情況下會影得更好？",
     "TARGET: 中影夾雜 -> 中英夾雜 (hot-word); 影得 homophones left as-is if unsure"),
    ("幫我 send 個 email 俾 David，話俾佢知我聽日唔得閒。",
     "REGRESSION: must stay identical"),
    ("呃我想睇下聽日個天氣點樣。",
     "REGRESSION: only filler 呃 removed"),
    ("這些使用背後的LLM模型都可以避免吧。",
     "REGRESSION: no reordering"),
]


def main():
    cfg = Config()
    vocab = vocab_terms(cfg.get("dictionary"), cfg.get("hotwords"))
    hk = cfg.get("traditional_hk")
    print(f"engine={cfg.get('asr_engine')} model={cfg.get('vllm_model')} "
          f"hotwords={vocab}\n")

    # ---- Part A: real SenseVoice transcription of the last recording --------
    wav = paths.AUDIO_TMP
    if os.path.exists(wav):
        asr = ASREngine(cfg.get("asr_engine"))
        asr.ensure_loaded()
        text = asr.transcribe(wav, cfg.get("language"), context=vocab)
        print(f"[Part A] SenseVoice live transcript ({asr.engine_name}):")
        print(f"  {text!r}\n")
    else:
        print("[Part A] no last_recording.wav found; skipping live ASR\n")

    # ---- Part B: full Clean post-correction pipeline, real GB10 vLLM --------
    llm = VLLMBackend(
        cfg.get("vllm_url"), cfg.get("vllm_model"),
        connect_timeout=cfg.get("vllm_connect_timeout"),
        ttft_timeout=cfg.get("vllm_ttft_timeout"),
        total_timeout=cfg.get("vllm_total_timeout"),
        api_key=(os.environ.get("WHISPERFLOW_VLLM_API_KEY")
                 or cfg.get("vllm_api_key")),
        reasoning_effort=cfg.get("vllm_reasoning_effort"),
    )
    if not llm.ping():
        print("[Part B] GB10 vLLM unreachable — aborting"); sys.exit(1)

    print("[Part B] Clean pipeline (quick_clean -> phonetic hotwords -> "
          "LLM edits -> apply_edits):\n")
    ok = True
    for raw, note in FIELD:
        base = quick_clean(raw, vocab=vocab, hk=hk)
        after_hw = apply_phonetic_hotwords(base, vocab)
        edits = llm.propose_edits(after_hw, vocab=vocab)
        final = apply_edits(after_hw, edits, vocab=vocab)
        print(f"  # {note}")
        print(f"  before: {raw}")
        print(f"  hotword:{after_hw}")
        print(f"  edits : {edits if edits else 'none'}")
        print(f"  after : {final}\n")
        # Cheap invariants surfaced for the transcript/evaluator.
        if "中影夾雜" in raw and "中英夾雜" not in final:
            print("  !! FAIL: 中英夾雜 not recovered"); ok = False
        if raw.startswith("幫我 send") and final != raw:
            print("  !! FAIL: regression sentence changed"); ok = False
    print("RESULT:", "ALL INVARIANTS HELD" if ok else "SOME INVARIANTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
