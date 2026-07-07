# WhisperFlow-Local — Full Product Specification (source of truth)

A **complete**, 100% local, offline Cantonese-English dictation app for macOS (Apple Silicon,
16GB RAM) — a full local clone of WhisperFlow / Wispr Flow. This document is the single source
of truth. "Done" = every acceptance criterion below is implemented for real and demonstrated.

## 0. Principles (non-negotiable)
- **100% local / offline.** The only network access allowed is (a) first-run model downloads and
  (b) `localhost` Ollama. No telemetry, no cloud ASR/LLM.
- **No mocks.** Zero fake / hardcoded / placeholder / stubbed transcripts or LLM outputs anywhere.
  Every layer calls the real model. `grep -riE "mock|fake|dummy|stub|placeholder|hardcode|lorem|TODO"`
  over source returns nothing meaningful.
- **Never freezes the UI.** All capture / ASR / LLM work runs off the AppKit main thread.
- **Graceful degradation.** Mic missing, Ollama down, or model absent must degrade (e.g. raw ASR
  with a clear message), never crash.

## 1. Tech stack (pinned)
- Python 3.11+, macOS menu-bar app via **`rumps`**.
- Global hotkeys + keystroke synthesis via **`pynput`**.
- Mic capture via **`sounddevice`** (16 kHz mono).
- **ASR = SenseVoiceSmall** via **`funasr`** `AutoModel("iic/SenseVoiceSmall")` + `fsmn-vad`,
  auto-language + inverse-text-normalization; emotion/event tags stripped from output.
  (Accepted equivalent if funasr install fails: `sherpa-onnx` SenseVoice int8.)
- **LLM = `qwen3.5:4b` via Ollama** (already installed), local API. Qwen3 is a hybrid reasoning
  model → **thinking mode MUST be disabled** (`think:false` / append `/no_think`) so it returns
  only the corrected text, no `<think>` blocks.
- **LLM backends (as of 2026-07-07):** the cleanup layer is now dual-backend — a remote vLLM
  server (`Qwen3.6-35B`, OpenAI `/v1`, streaming with a ~1 s time-to-first-token fallback) is
  primary, with the local Ollama `qwen3.5:4b` as automatic fallback and a 3-strike circuit
  breaker + cooldown auto-retry. `router.LLMRouter` composes them; menu → AI Model pins local.
  See `docs/superpowers/specs/2026-07-07-vllm-primary-llm-backend-design.md`. Thinking stays off
  on both.
- Clipboard via **`pyperclip`**. History via **SQLite** (stdlib `sqlite3`).
- Config as JSON under `~/Library/Application Support/WhisperFlow-Local/`.

## 2. Features & acceptance criteria

### A. Capture & hotkeys
- A1 **Push-to-talk**: hold a global hotkey (default Right Option) to record; release to transcribe.
- A2 **Hands-free toggle**: a second hotkey starts/stops recording without holding.
- A3 Hotkeys are **configurable** in Settings and persist across restarts.
- A4 Live **input-level / recording indicator** while capturing.
- AC: each mode records real audio and produces a real transcript inserted into the focused app.

### B. ASR engine
- B1 SenseVoiceSmall transcribes Cantonese, English, and **Cantonese-English code-switching** in
  one utterance without translating.
- B2 **VAD** handles multi-second / multi-sentence utterances.
- B3 **Language mode**: Auto / Cantonese / English / Mixed (selectable; Auto is default).
- B4 Model files download on first run with **visible progress**; cached afterwards (fully offline).
- AC: a real audio clip → real transcript printed (not asserted).

### C. AI formatting & commands (LLM layer, toggleable)
- C1 **Cleanup**: punctuation, capitalization, filler-word removal, keep code-switch verbatim.
- C2 **Formatting profiles** (each a distinct system prompt): `Raw` (no LLM), `Clean`, `Email`,
  `Message/Chat`, `Notes/Bullets`. User-selectable; default configurable.
- C3 **Voice commands** parsed from speech and applied, not typed literally: new line, new
  paragraph, "scratch that"/"delete that" (drop last segment), "all caps", "send"/"press enter".
- C4 **AI text commands** over the last dictation or current clipboard selection: "make this more
  formal", "summarize", "translate to English", "translate to Cantonese".
- AC: real qwen3.5:4b (thinking disabled) output shown for C1 and at least one C2 profile, one C3
  command, one C4 command.

### D. Personalization
- D1 **Custom dictionary**: user word/replacement list (names, jargon, spellings) applied as
  post-processing and/or prompt hints; editable in Settings.
- D2 **Per-app context**: detect the frontmost app (AppKit `NSWorkspace`) and auto-select a
  formatting profile via user-editable rules (e.g. Mail→Email, Messages/Slack→Chat, Terminal/VS Code→Raw).
- D3 All personalization persists in config.
- AC: a dictionary replacement demonstrably changes output; a per-app rule demonstrably switches profile.

### E. Text injection
- E1 Insert transcript into the focused app via clipboard + synthesized Cmd+V, then **restore the
  previous clipboard**.
- E2 **Fallback**: direct keystroke typing if paste fails.
- E3 **Copy-only** option (no auto-paste).
- AC: injection path exercised (documented manual check for cross-app paste + automated clipboard test).

### F. History
- F1 Persist every transcription (timestamp, raw text, formatted text, app, profile) in SQLite.
- F2 View recent history from the menu; copy or re-insert any entry; clear history.
- AC: after runs, DB contains real rows; re-insert works.

### G. Menu-bar UX, settings, onboarding
- G1 Menu-bar **status icon** reflects state: idle / recording / transcribing / formatting / error.
- G2 **Settings** UI (menu or window): hotkeys, ASR language, LLM on/off + model, default profile,
  dictionary editor, per-app rules, copy-only toggle, launch-at-login.
- G3 **Onboarding / first run**: guide the user to grant Microphone + Accessibility + Input
  Monitoring; show model-download progress.
- G4 **Feedback**: start/stop cue (sound or HUD) and a completion/error notification.
- AC: app launches, menu-bar item + menu items present (shown via run/log).

### H. Robustness & privacy
- H1 UI never blocks (threaded pipeline).
- H2 Ollama down → fall back to raw ASR with a clear message; model missing → actionable error.
- H3 No network calls except localhost Ollama + first-run downloads (documented; grep/report).
- AC: an Ollama-down path shown degrading gracefully (e.g. by pointing the client at a bad port in a test).

### I. Packaging
- I1 One-command run after `pip install -r requirements.txt` (documented entrypoint).
- I2 **Launch-at-login** option.
- I3 (Nice-to-have) `.app` bundle via py2app/pyinstaller with a build script.
- AC: entrypoint runs; launch-at-login toggles a real LaunchAgent/login item.

### J. Testing & verification
- J1 Real **end-to-end** script: real audio → real SenseVoiceSmall transcript → real qwen3.5:4b
  cleanup → printed output.
- J2 Automated tests for: voice-command parsing (C3), dictionary replacement (D1), per-app profile
  selection (D2), history storage (F1), tag-stripping (B1). All pass, output shown.
- J3 No-mocks grep (Principle 0) run and shown clean.
- J4 A **self-audit checklist** mapping every criterion A1…I3 to its implementing module + test,
  printed at the end.

## 3. Non-goals
- Cloud ASR/LLM, accounts, sync, mobile. Native-streaming ASR (chunked/VAD is acceptable).
  A perfectly signed/notarized distributable (unsigned local build is fine).

## 4. Definition of done
Every acceptance criterion in §2 is implemented for real and its evidence is surfaced in the
conversation (test output, real e2e stdout, clean grep, launch log, and the §J4 self-audit table).
