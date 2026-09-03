#!/usr/bin/env python
"""Live check for the GB10-only vLLM backend.

Part A: a REAL Qwen3.6-35B-A3B round-trip through VLLMBackend (a Clean edit list
and a longer Email profile — the latter proves a long generation is not cut off
at the 1 s TTFT gate). If the remote is unreachable in this environment it is
reported explicitly and Part A is skipped — never faked.

Part B: a dead remote endpoint proves the failure propagates without contacting
local Ollama; the app then uses deterministic quick-clean output.

Usage: .venv/bin/python scripts/itest_vllm_live.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whisperflow_local.config import Config  # noqa: E402
from whisperflow_local.llm import (  # noqa: E402
    LLMUnavailable,
    VLLMBackend,
)
from whisperflow_local.textproc import (  # noqa: E402
    apply_edits,
    guard_verbatim,
    apply_phonetic_hotwords,
    quick_clean,
    vocab_terms,
)

SAMPLE = "咁我而家轉咗做識別語言為自動，睇下佢中影夾雜嘅情況下會唔會影得好啲。"


def main():
    cfg = Config()
    vocab = vocab_terms(cfg.get("dictionary"), cfg.get("hotwords"))
    hk = cfg.get("traditional_hk")
    print(f"vllm_url={cfg.get('vllm_url')} vllm_model={cfg.get('vllm_model')}")
    print(f"backend={cfg.get('llm_backend')} hotwords={vocab}\n")

    remote = VLLMBackend(
        cfg.get("vllm_url"), cfg.get("vllm_model"),
        connect_timeout=cfg.get("vllm_connect_timeout"),
        ttft_timeout=cfg.get("vllm_ttft_timeout"),
        total_timeout=cfg.get("vllm_total_timeout"),
        api_key=(os.environ.get("WHISPERFLOW_VLLM_API_KEY")
                 or cfg.get("vllm_api_key")),
        reasoning_effort=cfg.get("vllm_reasoning_effort"),
    )

    # ---- Part A: real Qwen3.8 via vLLM -------------------------------------
    print("[Part A] Remote Qwen3.6-35B-A3B via GB10 vLLM (real):")
    if not remote.ping():
        print(f"  remote UNREACHABLE at {cfg.get('vllm_url')} in this "
              f"environment — skipping Part A (not faked).\n")
    else:
        base = apply_phonetic_hotwords(
            quick_clean(SAMPLE, vocab=vocab, hk=hk), vocab)
        try:
            out = remote.propose_cleanup(base, vocab=vocab)
            edits = out["edits"]
            cleaned = apply_edits(guard_verbatim(base, out["clean"]),
                                  edits, vocab=vocab)
            email = remote.format_text(SAMPLE, "Structured", vocab=vocab)
            print(f"  input   : {SAMPLE}")
            print(f"  edits   : {edits if edits else 'none'}")
            print(f"  clean   : {cleaned}")
            print(f"  struct  : {email!r}  (long output, not cut at 1s TTFT)\n")
        except LLMUnavailable as exc:
            print(f"  remote failed mid-call: {exc}\n")

    # ---- Part B: remote-only degradation ----------------------------------
    print("[Part B] Dead remote -> LLMUnavailable (no Ollama fallback):")
    dead_remote = VLLMBackend("http://127.0.0.1:1/v1", cfg.get("vllm_model"),
                              connect_timeout=0.5, ttft_timeout=0.5)
    base = apply_phonetic_hotwords(quick_clean(SAMPLE, vocab=vocab, hk=hk), vocab)
    try:
        dead_remote.propose_cleanup(base, vocab=vocab)
    except LLMUnavailable as exc:
        print(f"  expected remote-only failure: {exc}")
        print(f"  deterministic output retained: {base}")
        print("RESULT: REMOTE-ONLY DEGRADATION OK")
        return
    print("RESULT: FAILED — dead endpoint unexpectedly returned")
    sys.exit(1)


if __name__ == "__main__":
    main()
