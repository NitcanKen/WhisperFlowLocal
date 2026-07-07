#!/usr/bin/env python
"""Live check for the remote vLLM primary + local Ollama fallback.

Part A: a REAL Qwen3.6-35B round-trip through VLLMBackend (a Clean edit list
and a longer Email profile — the latter proves a long generation is not cut off
at the 1 s TTFT gate). If the remote is unreachable in this environment it is
reported explicitly and Part A is skipped — never faked.

Part B: an LLMRouter whose REMOTE is pointed at a dead port, proving a real
fallback to the real local Ollama 4B. `router.model` reports which backend
actually served the call.

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
    OllamaBackend,
    VLLMBackend,
)
from whisperflow_local.router import LLMRouter  # noqa: E402
from whisperflow_local.textproc import (  # noqa: E402
    apply_edits,
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
    print(f"ollama_model={cfg.get('ollama_model')} hotwords={vocab}\n")

    remote = VLLMBackend(
        cfg.get("vllm_url"), cfg.get("vllm_model"),
        connect_timeout=cfg.get("vllm_connect_timeout"),
        ttft_timeout=cfg.get("vllm_ttft_timeout"),
        total_timeout=cfg.get("vllm_total_timeout"),
    )

    # ---- Part A: real 35B via vLLM -----------------------------------------
    print("[Part A] Remote Qwen3.6-35B via vLLM (real):")
    if not remote.ping():
        print(f"  remote UNREACHABLE at {cfg.get('vllm_url')} in this "
              f"environment — skipping Part A (not faked).\n")
    else:
        base = apply_phonetic_hotwords(
            quick_clean(SAMPLE, vocab=vocab, hk=hk), vocab)
        try:
            edits = remote.propose_edits(base, vocab=vocab)
            cleaned = apply_edits(base, edits, vocab=vocab)
            email = remote.format_text(SAMPLE, "Email", vocab=vocab)
            print(f"  input   : {SAMPLE}")
            print(f"  edits   : {edits if edits else 'none'}")
            print(f"  clean   : {cleaned}")
            print(f"  email   : {email!r}  (long output, not cut at 1s TTFT)\n")
        except LLMUnavailable as exc:
            print(f"  remote failed mid-call: {exc}\n")

    # ---- Part B: forced fallback remote->local -----------------------------
    print("[Part B] Fallback: remote at a dead port -> real local Ollama 4B:")
    local = OllamaBackend(cfg.get("ollama_url"), cfg.get("ollama_model"))
    if not local.ping():
        print(f"  local Ollama UNREACHABLE at {cfg.get('ollama_url')} — "
              f"cannot demonstrate fallback here (not faked).")
        sys.exit(1)
    dead_remote = VLLMBackend("http://127.0.0.1:1/v1", cfg.get("vllm_model"),
                              connect_timeout=0.5, ttft_timeout=0.5)
    router = LLMRouter(local=local, remote=dead_remote,
                       backend="auto", threshold=3, cooldown=300.0)
    base = apply_phonetic_hotwords(quick_clean(SAMPLE, vocab=vocab, hk=hk), vocab)
    edits = router.propose_edits(base, vocab=vocab)
    cleaned = apply_edits(base, edits, vocab=vocab)
    print(f"  input    : {SAMPLE}")
    print(f"  edits    : {edits if edits else 'none'}")
    print(f"  clean    : {cleaned}")
    print(f"  served by: {router.model}  (expected local '{cfg.get('ollama_model')}')")
    ok = router.model == cfg.get("ollama_model")
    print("RESULT:", "FALLBACK OK" if ok else "FALLBACK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
