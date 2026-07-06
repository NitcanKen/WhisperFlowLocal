#!/usr/bin/env python
"""End-to-end dictation latency benchmark.

Measures wall time from *recording stop* to *text output complete*:
  save_wav -> ASR -> dictionary/hotwords -> voice commands
  -> LLM formatting (Ollama, default profile) -> insertion (test sink =
  clipboard-only path, so no window focus is needed).

Models are warmed (one throwaway run per configuration) before timing.

Budget: the default config always runs the LLM correction pass (the old
disfluency fast-path was removed so ASR homophone slips actually get fixed),
so it is LLM-bound at ~2.5-3 s — the deterministic ASR+hotword+quick_clean
work is ~0.45 s (see the "raw (no LLM)" row, which is the sub-second path a
user gets by toggling the LLM off). SLO below tracks the default (LLM-on).

Usage: .venv/bin/python scripts/bench_latency.py [--runs 10] [--wav PATH]
"""
import argparse
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf  # noqa: E402

from whisperflow_local.asr import ASREngine  # noqa: E402
from whisperflow_local.config import Config  # noqa: E402
from whisperflow_local.injector import insert  # noqa: E402
from whisperflow_local.llm import LLMClient, LLMUnavailable  # noqa: E402
from whisperflow_local.textproc import (  # noqa: E402
    apply_dictionary,
    apply_edits,
    apply_phonetic_hotwords,
    parse_voice_commands,
    quick_clean,
    vocab_terms,
)

DEFAULT_WAV = os.path.join(os.path.dirname(__file__), "..",
                           "tests", "fixtures", "yue_en_5s.wav")
SLO_S = 3.0  # default is LLM-bound; 'raw (no LLM)' is the sub-second path


def pipeline_once(audio, asr, cfg, llm, mode):
    """One full recording-stop -> output-complete pass, mirroring
    app._process_audio routing. Returns (dt, text).

    mode: "default" (what the app ships: LLM Clean, quick_clean on failure)
          | "raw"   (no LLM, no cleanup)
    """
    from whisperflow_local import paths
    from whisperflow_local.audio import save_wav

    vocab = vocab_terms(cfg.get("dictionary"), cfg.get("hotwords"))
    hk = cfg.get("traditional_hk")
    t0 = time.perf_counter()
    wav = save_wav(audio, paths.AUDIO_TMP)
    raw = asr.transcribe(wav, cfg.get("language"), context=vocab)
    text = apply_dictionary(raw, cfg.get("dictionary"))
    parsed = parse_voice_commands(text)
    formatted = parsed.text
    if mode == "default":
        formatted = quick_clean(formatted, vocab=vocab, hk=hk)
        formatted = apply_phonetic_hotwords(formatted, vocab)
        try:
            edits = llm.propose_edits(formatted, vocab=vocab)
            formatted = apply_edits(formatted, edits, vocab=vocab)
        except LLMUnavailable:
            pass  # quick_clean + hotword output already stands on its own
    insert(formatted, copy_only=True)  # test sink: clipboard, no focus needed
    return time.perf_counter() - t0, formatted


def bench(label, audio, asr, cfg, llm, mode, runs):
    pipeline_once(audio, asr, cfg, llm, mode)  # warmup (not timed)
    times, last_text = [], ""
    for _ in range(runs):
        dt, last_text = pipeline_once(audio, asr, cfg, llm, mode)
        times.append(dt)
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, int(round(0.95 * len(times))) - 1)]
    return {"label": label, "p50": p50, "p95": p95,
            "min": min(times), "max": max(times), "text": last_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--wav", default=DEFAULT_WAV)
    ap.add_argument("--engines", default="sensevoice,qwen3")
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000, "fixture must be 16 kHz mono (matches Recorder)"
    print(f"fixture: {args.wav} ({len(audio)/sr:.2f}s @ {sr}Hz), "
          f"runs per config: {args.runs}")

    cfg = Config()
    llm = LLMClient(cfg.get("ollama_url"), cfg.get("ollama_model"))
    if not llm.ping():
        print("WARNING: Ollama unreachable — LLM rows will be skipped")

    rows = []
    for engine_name in args.engines.split(","):
        asr = ASREngine(engine_name)
        asr.ensure_loaded()
        rows.append(bench(f"{engine_name} default (LLM Clean)", audio,
                          asr, cfg, llm, mode="default", runs=args.runs))
        rows.append(bench(f"{engine_name} raw (no LLM)", audio,
                          asr, cfg, llm, mode="raw", runs=args.runs))

    print(f"\n{'configuration':34s} {'p50':>7s} {'p95':>7s} "
          f"{'min':>7s} {'max':>7s}  SLO(p50<={SLO_S}s)")
    for r in rows:
        ok = "PASS" if r["p50"] <= SLO_S else "FAIL"
        print(f"{r['label']:34s} {r['p50']:6.2f}s {r['p95']:6.2f}s "
              f"{r['min']:6.2f}s {r['max']:6.2f}s  {ok}")
    for r in rows:
        print(f"output [{r['label']}]: {r['text']!r}")
    default_row = rows[0]  # sensevoice default = the shipped configuration
    sys.exit(0 if default_row["p50"] <= SLO_S else 1)


if __name__ == "__main__":
    main()
