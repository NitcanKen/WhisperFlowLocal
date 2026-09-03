# WhisperFlow-Local

Private Cantonese-English dictation menu-bar app for macOS (Apple Silicon).
Hold a key → record → ASR (GB10 Qwen3-ASR primary + local SenseVoice fallback)
→ LLM cleanup (GB10 Qwen3.8 vLLM only; no Ollama fallback) → paste into the
focused app. Python 3.11,
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
  asr.py        # ASR backends: SenseVoiceEngine (local, funasr) + RemoteQwenASRBackend (remote vLLM /v1/audio/transcriptions); Qwen3ASREngine (local on-device, off-menu, kept for tests)
  asr_router.py # ASRRouter: remote Qwen3-ASR primary + local SenseVoice fallback + breaker (mirrors router.py)
  llm.py        # LLM backends: OllamaBackend (local) + VLLMBackend (remote, streaming/TTFT); cleanup/structured/generation prompts, thinking off
  clarify.py    # clarify panel for generation mode: keyable non-activating NSPanel + worker/main handoff
  router.py     # LLMRouter: remote-primary + local fallback + 3-strike breaker + cooldown retry
  breaker.py    # CircuitBreaker: 3-strike trip + cooldown re-probe, shared by both routers
  textproc.py   # voice commands, custom dictionary, punctuation, OpenCC s2hk, guard_verbatim
  hotkeys.py keycap.py  # global hold bindings (dictate / Shift+PTT = generate) + toggle (pynput)
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
- IMPORTANT: `Verbatim` DOES send a full-sentence rewrite (an edit list can't move
  the ASR's misplaced 。／？), but the result is only used if it passes
  `textproc.guard_verbatim` — output characters minus punctuation/case must be a
  **subsequence** of the input's and keep ≥70%. That makes translation, reordering,
  書面語 conversion and summarising impossible while allowing filler deletion and
  repunctuation. Never drop the guard, and never widen it into a similarity score;
  unguarded rewriting corrupts Cantonese and code-switching. `Structured` is
  deliberately unguarded — it IS a transformation.
- Exactly two formatting modes (`Verbatim`, `Structured`), chosen from the menu and
  never overridden. There are no per-app rules and no `Raw`.
- All user settings flow through `config.Config` / `DEFAULTS`; persist via `.set()`.
- IMPROTANT: Whenever need to explore or understand the codebase, use codebase-memory skill.

## Gotchas
- Never run capture / ASR / LLM on the AppKit main thread — the menu-bar UI must
  never freeze. Keep the pipeline on worker threads. The clarify round inverts
  this deliberately: the WORKER blocks on `ClarifyRequest.done` while the main
  thread drives the panel from `_refresh_ui`. Both waits are bounded, so neither
  side can deadlock.
- `_finish_clarify` must hide the panel BEFORE resolving the request. The worker
  synthesizes ⌘V the moment it resumes, and a still-key panel would eat the paste.
- pynput has NO `fn` key on macOS (50 keys, none is `fn`), and `_on_press` records
  the pressed key's own modifier in `_mods_held` *before* matching — so a bare
  Right-Option press arrives as `{"alt"}`, never `set()`. Hold bindings compare
  `context_mods`, not the raw set.
- Degrade, never crash: GB10 LLM down → deterministic cleanup/raw ASR with a
  notice, never Ollama; mic/model missing → actionable error.
- After `make_app.sh` rebuilds the .app, macOS TCC can silently deny
  Accessibility / Input Monitoring even though System Settings still shows them
  ON (the binary's cdhash changed). The script resets stale grants; if paste or
  hotkeys die right after a rebuild, re-grant and restart. Don't work around it
  in code.
- The .app runs a compiled `launcher.c` shim that execs the venv Python against
  the **live repo**, so just relaunching picks up `.py` edits — no rebuild, no
  cdhash change. Only run `make_app.sh` when `launcher.c`, `Info.plist`, or the
  icon change; needlessly rebuilding is what triggers the TCC breakage above.
