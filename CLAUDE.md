# WhisperFlow-Local

100% local, offline Cantonese-English dictation menu-bar app for macOS (Apple
Silicon). Hold a key → record → SenseVoiceSmall ASR → LLM cleanup (remote vLLM
primary, local Ollama fallback) → paste into the focused app. Python 3.11,
`rumps` menu bar. `SPEC.md` is the
source of truth (acceptance criteria A1–J4) — read it before adding features.

## Commands
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m whisperflow_local             # run the app (dev)
.venv/bin/python -m whisperflow_local --selftest  # build menu/config, no GUI or model load
.venv/bin/python -m pytest tests/                 # unit tests
.venv/bin/python -m pytest tests/test_textproc.py # single test file
.venv/bin/python scripts/e2e.py                   # real audio → real ASR → real LLM
scripts/make_app.sh                               # build ~/Applications/WhisperFlow-Local.app
```

## Structure
```
whisperflow_local/
  app.py        # rumps menu-bar app + record→ASR→LLM→inject pipeline (run via __main__)
  asr.py        # SenseVoiceSmall (funasr); optional qwen3 engine, falls back to sensevoice
  llm.py        # LLM backends: OllamaBackend (local) + VLLMBackend (remote, streaming/TTFT); edit-list prompts, thinking off
  router.py     # LLMRouter: remote-primary + local fallback + 3-strike breaker + cooldown retry
  textproc.py   # voice commands, custom dictionary, punctuation, OpenCC s2hk
  hotkeys.py keycap.py  # global push-to-talk + toggle hotkeys (pynput)
  injector.py   # clipboard + synthesized ⌘V paste, then restore prior clipboard
  overlay.py    # native waveform HUD shown while recording
  config.py paths.py    # JSON config + ~/Library/Application Support/WhisperFlow-Local
  history.py i18n.py permissions.py launchagent.py frontmost.py audio.py
scripts/  # make_app.sh, build_app.sh, e2e.py, live itest_*.py, bench_latency.py
tests/    # pytest; conftest isolates app.log from the real one
```

## Conventions
- IMPORTANT: No mocks in product code. No fake/stub/placeholder transcripts or
  LLM outputs anywhere under `whisperflow_local/`; every layer calls the real
  model. `grep -riE "mock|fake|dummy|stub|placeholder|hardcode|lorem|TODO"` over
  source must stay clean. Tests and `e2e.py` exercise the real ASR/LLM (or real
  error paths), never fakes.
- IMPORTANT: Keep Qwen thinking mode off — `think: false` on Ollama,
  `chat_template_kwargs.enable_thinking: false` on vLLM. Enabling it adds
  60–180 s/utterance and leaks `<think>` blocks into the output.
- Clean profile uses guarded edit-list prompts (see `llm.py`), not full-sentence
  rewriting — rewriting corrupts Cantonese and code-switching. Don't collapse it
  back into a rewrite prompt.
- All user settings flow through `config.Config` / `DEFAULTS`; persist via `.set()`.
- IMPROTANT: Whenever need to explore or understand the codebase, use codebase-memory skill.

## Gotchas
- Never run capture / ASR / LLM on the AppKit main thread — the menu-bar UI must
  never freeze. Keep the pipeline on worker threads.
- Degrade, never crash: Ollama down → paste raw ASR with a notice; mic/model
  missing → actionable error.
- After `make_app.sh` rebuilds the .app, macOS TCC can silently deny
  Accessibility / Input Monitoring even though System Settings still shows them
  ON (the binary's cdhash changed). The script resets stale grants; if paste or
  hotkeys die right after a rebuild, re-grant and restart. Don't work around it
  in code.
- The .app runs a compiled `launcher.c` shim that execs the venv Python against
  the **live repo**, so just relaunching picks up `.py` edits — no rebuild, no
  cdhash change. Only run `make_app.sh` when `launcher.c`, `Info.plist`, or the
  icon change; needlessly rebuilding is what triggers the TCC breakage above.
