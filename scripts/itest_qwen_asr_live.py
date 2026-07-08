#!/usr/bin/env python
"""Live check for the remote Qwen3-ASR primary + local SenseVoice fallback.

Part A: a REAL transcription of tests/fixtures/yue_en_5s.wav through
RemoteQwenASRBackend, with the vocab/hotword list passed as the biasing prompt.
If the remote is unreachable — or reachable but not yet audio-capable (needs
`pip install "vllm[audio]"` on the box) — that is reported explicitly and Part A
is skipped; nothing is faked.

Part B: an ASRRouter whose REMOTE is pointed at a dead port, proving a real
fallback to the real local SenseVoice. `router.engine_name` reports which engine
actually served the utterance.

Usage: .venv/bin/python scripts/itest_qwen_asr_live.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whisperflow_local.asr import (  # noqa: E402
    ASRUnavailable,
    RemoteQwenASRBackend,
    SenseVoiceEngine,
)
from whisperflow_local.asr_router import ASRRouter  # noqa: E402
from whisperflow_local.config import Config  # noqa: E402
from whisperflow_local.textproc import vocab_terms  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..",
                       "tests", "fixtures", "yue_en_5s.wav")


def main():
    cfg = Config()
    vocab = vocab_terms(cfg.get("dictionary"), cfg.get("hotwords"))
    language = cfg.get("language")
    print(f"qwen_asr_url={cfg.get('qwen_asr_url')} "
          f"model={cfg.get('qwen_asr_model')}")
    print(f"language={language} hotwords={vocab}\n")

    remote = RemoteQwenASRBackend(
        cfg.get("qwen_asr_url"), cfg.get("qwen_asr_model"),
        connect_timeout=cfg.get("qwen_asr_connect_timeout"),
        total_timeout=cfg.get("qwen_asr_total_timeout"),
    )

    # ---- Part A: real remote Qwen3-ASR -------------------------------------
    print("[Part A] Remote Qwen3-ASR via vLLM (real):")
    if not remote.ping():
        print(f"  remote UNREACHABLE at {cfg.get('qwen_asr_url')} in this "
              f"environment — skipping Part A (not faked).\n")
    else:
        try:
            text = remote.transcribe(FIXTURE, language, context=vocab)
            print(f"  fixture : {os.path.basename(FIXTURE)}")
            print(f"  remote  : {text!r}\n")
        except ASRUnavailable as exc:
            print(f"  remote reachable but transcription failed: {exc}")
            print("  (if this is a decode error, the box needs "
                  "`pip install \"vllm[audio]\"` + restart.)\n")

    # ---- Part B: forced fallback remote->local SenseVoice ------------------
    print("[Part B] Fallback: remote at a dead port -> real local SenseVoice:")
    local = SenseVoiceEngine()
    dead_remote = RemoteQwenASRBackend("http://127.0.0.1:1/v1",
                                       cfg.get("qwen_asr_model"),
                                       connect_timeout=0.5, total_timeout=0.5)
    router = ASRRouter(local=local, remote=dead_remote, backend="auto",
                       threshold=3, cooldown=300.0)
    print("  loading SenseVoice (downloads on first run)…")
    router.ensure_loaded()
    text = router.transcribe(FIXTURE, language, context=vocab)
    print(f"  fixture  : {os.path.basename(FIXTURE)}")
    print(f"  text     : {text!r}")
    print(f"  served by: {router.engine_name}  (expected 'sensevoice')")
    ok = bool(router.engine_name == "sensevoice" and text.strip())
    print("RESULT:", "FALLBACK OK" if ok else "FALLBACK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
