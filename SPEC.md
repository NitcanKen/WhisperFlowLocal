# WhisperFlow-Local — Full Product Specification (source of truth)

A **complete**, 100% local, offline Cantonese-English dictation app for macOS (Apple Silicon,
16GB RAM) — a full local clone of WhisperFlow / Wispr Flow. This document is the single source
of truth. "Done" = every acceptance criterion below is implemented for real and demonstrated.

## 0. Principles (non-negotiable)
- **Private self-hosted inference.** Network model traffic is limited to the
  user's GB10 over Tailscale plus first-run model downloads. No telemetry or
  third-party cloud ASR/LLM.
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
- **ASR backends (as of 2026-07-08):** the recognizer is dual-backend, mirroring the LLM — a
  remote **Qwen3-ASR** (vLLM OpenAI `/v1/audio/transcriptions`, port 8001) is primary with the
  local SenseVoice as automatic fallback + the same 3-strike breaker/cooldown (shared
  `breaker.CircuitBreaker`). `asr_router.ASRRouter` composes them; menu → ASR Engine pins local.
  Hotwords (`vocab_terms`) bias the remote natively via the request `prompt` and still drive
  `apply_phonetic_hotwords` for SenseVoice. See
  `docs/superpowers/specs/2026-07-08-qwen-asr-remote-backend-design.md`.
- **LLM = `Qwen3.6-35B-A3B` via vLLM on the private GB10.** Thinking mode MUST be
  disabled so it returns only the corrected text, with no `<think>` blocks.
- **LLM backend (as of 2026-08-27):** `llm_backend="remote"` sends every cleanup
  request to the GB10 OpenAI `/v1` endpoint. Remote failure propagates to the
  app's deterministic cleanup path; local Ollama is never contacted. Legacy
  backend classes remain only for compatibility with older configs/tests.
- Clipboard via **`pyperclip`**. History via **SQLite** (stdlib `sqlite3`).
- Config as JSON under `~/Library/Application Support/WhisperFlow-Local/`.

## 2. Features & acceptance criteria

### A. Capture & hotkeys
- A1 **Push-to-talk**: hold a global hotkey (default Right Option) to record; release to transcribe.
- A2 **Hands-free toggle**: a second hotkey starts/stops recording without holding.
- A2b **Generation hold**: `Shift` + the push-to-talk key records a *content-generation*
  request instead (C4). Both are HELD bindings matched by one predicate on the modifier
  *context* (the pressed key's own modifier family is excluded), so they are mutually
  exclusive by construction. `fn` is NOT usable: pynput exposes no `fn` key on macOS.
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

### C. AI formatting & generation (LLM layer, toggleable)
- C1 **Cleanup**: punctuation repair, filler-word and stutter removal, ASR homophone
  correction; code-switching kept verbatim.
- C2 **Two formatting modes** (as of 2026-09-03), picked from the menu and never
  overridden by anything else:
  - `Verbatim` (原文口語) — keeps the spoken wording. The model rewrites the whole
    utterance (an edit list cannot move a misplaced 。), but every rewrite must pass
    `textproc.guard_verbatim`: the output's characters minus punctuation and case must
    be a **subsequence** of the input's and keep ≥70% of them. That permits deleting
    fillers and rewriting marks while making translation, reordering, 書面語 conversion
    and summarising impossible. Homophone fixes travel as a separate edit list through
    `textproc.apply_edits`. Both channels come back from ONE model call.
  - `Structured` (書面結構化) — understands the utterance and re-emits it as structured
    written Chinese: 口語→書面語, self-corrections resolved to the speaker's final
    intent, grouped and numbered. Deliberately unguarded; it is a transformation.
- C3 **Voice commands** parsed from speech and applied, not typed literally: new line, new
  paragraph, "scratch that"/"delete that" (drop last segment), "all caps", "send"/"press enter".
- C4 **Content generation** (hold `Shift`+push-to-talk): the user speaks a *request* and the
  model writes the content, which is pasted like a dictation. A vague request first gets a
  **clarify panel** (≤2 questions, 2-3 options each plus free text) at the waveform pill's
  anchor, answered by click or by pressing 1-3, Esc to cancel. The panel is a nonactivating
  NSPanel that becomes key WITHOUT activating the app, so the paste target is preserved.
- AC: real GB10 output shown for both C2 modes, one C3 command, and both C4 paths
  (clear request → written directly; vague request → clarified, then written).

### D. Personalization
- D1 **Custom dictionary**: user word/replacement list (names, jargon, spellings) applied as
  post-processing and/or prompt hints; editable in Settings.
- D2 **(removed 2026-09-03)** Per-app formatting rules. They silently overrode the mode the
  user had ticked in the menu, which is the opposite of the intended contract: the menu
  choice is authoritative. `Config.load` migrates old `app_rules` away.
- D3 All personalization persists in config.
- AC: a dictionary replacement demonstrably changes output.

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
- G2 **Settings** UI (menu or window): hotkeys, ASR language, LLM on/off + model, formatting
  mode (2 options), dictionary editor, copy-only toggle, launch-at-login.
- G3 **Onboarding / first run**: guide the user to grant Microphone + Accessibility + Input
  Monitoring; show model-download progress.
- G4 **Feedback**: start/stop cue (sound or HUD) and a completion/error notification.
- AC: app launches, menu-bar item + menu items present (shown via run/log).

### H. Robustness & privacy
- H1 UI never blocks (threaded pipeline).
- H2 GB10 LLM down → deterministic cleanup/raw ASR with a clear message; model
  missing → actionable error; never call Ollama.
- H3 Model network calls are limited to private GB10 endpoints + first-run downloads.
- AC: a dead-GB10 path shown degrading gracefully without an Ollama fallback.

### I. Packaging
- I1 One-command run after `pip install -r requirements.txt` (documented entrypoint).
- I2 **Launch-at-login** option.
- I3 (Nice-to-have) `.app` bundle via py2app/pyinstaller with a build script.
- AC: entrypoint runs; launch-at-login toggles a real LaunchAgent/login item.

### J. Testing & verification
- J1 Real **end-to-end** script: real audio → real SenseVoiceSmall transcript →
  real GB10 Qwen3.8 cleanup → printed output.
- J2 Automated tests for: voice-command parsing (C3), dictionary replacement (D1), the
  Verbatim rewrite guard (C2 — accepts filler/punctuation edits, rejects translation,
  reordering, insertion and summarising), clarify parsing + the worker/main handoff (C4),
  hold-binding exclusivity (A2b), history storage (F1), tag-stripping (B1). All pass.
- J3 No-mocks grep (Principle 0) run and shown clean.
- J4 A **self-audit checklist** mapping every criterion A1…I3 to its implementing module + test,
  printed at the end.

## 3. Non-goals
- Cloud ASR/LLM, accounts, sync, mobile. Native-streaming ASR (chunked/VAD is acceptable).
  A perfectly signed/notarized distributable (unsigned local build is fine).

## 4. Definition of done
Every acceptance criterion in §2 is implemented for real and its evidence is surfaced in the
conversation (test output, real e2e stdout, clean grep, launch log, and the §J4 self-audit table).
