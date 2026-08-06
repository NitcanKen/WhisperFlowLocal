# Remote Qwen3-ASR primary ASR backend with local SenseVoice fallback — Design

Date: 2026-07-08
Status: Approved (design), ready for implementation

## 1. Context & motivation

Today the ASR layer (`whisperflow_local/asr.py`) runs entirely on-device: an
`ASREngine` facade routes to `SenseVoiceEngine` (funasr, ~1 GB, CPU) or a local
`Qwen3ASREngine` (`qwen_asr` package, `Qwen/Qwen3-ASR-1.7B`, ~4 GB on MPS), the
latter falling back to SenseVoice if it fails to load.

The user now serves **the same `Qwen/Qwen3-ASR-1.7B` via vLLM** on a separate GPU
box at a private endpoint such as `http://asr-host.example:8001/v1`. During
development, `GET /v1/models` returned that id (`max_model_len` 65536) and
`POST /v1/audio/transcriptions` is registered (`allow: POST`).

Goal — mirror the LLM router already shipped (`router.py` + `VLLMBackend`): make the
**remote Qwen3-ASR the primary** recognizer with the **local SenseVoice as an
automatic fallback** when the remote is slow/unreachable, plus a **circuit breaker**
so a persistently-down remote stops costing latency on every dictation. Selecting
the local engine must **never** contact the remote until re-selected.

### 1a. Server-side prerequisite (external blocker, not part of this change)

The remote vLLM was started **without audio extras**, so it currently cannot decode
audio: `/v1/audio/transcriptions` returns `"Invalid or unsupported audio file"`
(even for a clean 16 kHz mono PCM wav and a re-encoded FLAC), and
`/v1/chat/completions` with `input_audio` returns
`"Please install vllm[audio] for audio support"`. The operator must
`pip install "vllm[audio]"` (pulls `librosa`/`soundfile`) and restart vLLM on
the private host. The client in this design is built and unit-tested regardless; the
**live** round-trip (AC1/AC3) is verified once the box is fixed. This is the one
external dependency and is subject to the §14 blocker rule.

## 2. Goals / non-goals

**Goals**
- Remote Qwen3-ASR (OpenAI `/v1/audio/transcriptions`) is primary in `auto` mode;
  local SenseVoice is fallback.
- Per-request fallback on remote trouble: a short **connect** timeout (~1 s)
  fast-fails an unreachable/asleep box (the common Tailscale-down case), plus a
  **generous read/total** deadline so a legitimately-working transcription is never
  cut off. (Same philosophy the user approved for the LLM; no hard total-1s.)
- Circuit breaker: after **3 consecutive fallbacks**, pin to local; after a
  **cooldown (default 5 min)** auto-retry the remote and switch back on success.
- Menu lets the user pin **local-only** (SenseVoice), which never touches the remote.
- **One hotword list, two mechanisms:** the existing `vocab_terms(dictionary,
  hotwords)` biases the remote recognizer natively (at decode time) *and* the
  existing `apply_phonetic_hotwords` post-hoc recovery still runs for the SenseVoice
  path — no new UI, no second list.
- **Memory:** remote holds zero local model memory, so SenseVoice stays resident as
  the instant fallback and **no 4 GB MPS load ever** happens in `auto` mode.

**Non-goals**
- No change to the on-device `Qwen3ASREngine` *class* or its tests; it is only
  **removed from the engine menu** (2-mode menu, mirroring the LLM's auto/local).
- No change to VAD, capture, injection, hotkeys, history, or the SenseVoice wire.
- Optional Bearer authentication is supported; transport security remains the
  endpoint operator's responsibility.
- No streaming of partial transcripts into the target app (we paste once, whole).

## 3. Architecture (base engines + remote backend + router)

Parallel to the LLM side (`llm.py` backends + `router.py` router):

```
asr.py
  SenseVoiceEngine            # unchanged — local resident fallback (name="sensevoice")
  Qwen3ASREngine              # unchanged class (kept for tests; off the menu)
  RemoteQwenASRBackend        # NEW: POST /v1/audio/transcriptions (name="qwen3")
    .transcribe(wav, language, context) -> str
    .ping() -> bool
  ASRUnavailable(Exception)   # NEW: raised by the remote backend on any failure

asr_router.py                 # NEW
  ASRRouter                   # same call surface app.py already uses on self.asr:
    .transcribe(wav, language="auto", context=None) -> str
    .ensure_loaded(progress_cb=None)      # loads SenseVoice (the resident fallback)
    .engine_name (property)               # who served the LAST call (mid-call log)
    .set_engine(code)                     # "qwen3"->auto, "sensevoice"->local
    .set_remote(url, model)               # (parity with LLM; config edits)

breaker.py                    # NEW (shared): CircuitBreaker used by BOTH routers
```

**Shared `CircuitBreaker` (refactor, user-approved).** The 3-strike trip, cooldown
re-probe, and injectable clock are identical in both routers. Extract them into a
tiny `breaker.CircuitBreaker`; `LLMRouter` is refactored to compose it while keeping
its public API byte-identical (guarded by `test_llm_router.py`), and `ASRRouter`
composes the same class. Net: both routers get thinner. See §5.

**Call-site compatibility.** `app.py` builds `self.asr = ASRRouter(...)` instead of
`ASREngine(...)`. Every existing call — `self.asr.ensure_loaded(...)`,
`self.asr.transcribe(wav, language, context=...)`, `self.asr.engine_name`,
`self.asr.set_engine(code)` — is preserved by the router's surface, so
`_process_audio` and `_preload` are unchanged.

## 4. Remote backend wire format (`RemoteQwenASRBackend`)

- Endpoint: `POST {qwen_asr_url}/audio/transcriptions` (`qwen_asr_url` ends in `/v1`).
- Body (multipart/form-data):
  - `file`: the wav bytes read from `wav_path` (16 kHz mono PCM, as saved today).
  - `model`: `qwen_asr_model`.
  - `language`: mapped via the existing `QWEN_LANGUAGE_MAP` (`auto/mixed`→omit for
    auto-detect; `yue`→`Cantonese`; `en`→`English`).
  - `prompt`: `", ".join(context)` — the hotword/vocab terms, for **native decode-time
    biasing** (the strong path; OpenAI-transcription `prompt` field).
  - `response_format`: `json`.
- Timeout: `requests.post(..., timeout=(qwen_asr_connect_timeout,
  qwen_asr_total_timeout))`. Connect (~1 s) fast-fails an unreachable box; read
  (~30 s) is the generous work deadline for a one-shot transcription.
- Success: parse `{"text": "..."}` → return `text.strip()`.
- Any of {connect error, read timeout, HTTP != 2xx, missing/malformed body} →
  raise **`ASRUnavailable`** (the router's fallback trigger). Never returns a
  fabricated transcript.
- `ping()`: `GET {qwen_asr_url}/models`, short timeout → bool.
- When `qwen_asr_api_key` is configured, both requests include
  `Authorization: Bearer <token>`.

> Verification caveat (§1a): the exact biasing field could not be exercised live
> because the box can't decode audio yet. `prompt` is the OpenAI-standard field vLLM
> forwards; if the live test shows Qwen3-ASR wants it under `extra_body`/`asr_options`
> instead, adjust the one request-builder line and re-verify. Everything else is
> unaffected.

## 5. Shared circuit breaker (`breaker.py`)

```python
class CircuitBreaker:
    def __init__(self, *, threshold=3, cooldown=300.0, clock=time.monotonic): ...
    def allow_remote(self) -> bool           # False only while tripped AND cooling
    def record_success(self) -> bool         # returns was_tripped (for "reconnected")
    def record_failure(self) -> bool         # returns tripped_now (for "fallback")
    def reset(self) -> None                   # on manual backend switch
```

Internals (lock-guarded, `_consecutive`, `_temporary`, `_retry_at`) are exactly the
logic in today's `LLMRouter._should_try_remote/_on_remote_success/_on_remote_failure`.
`LLMRouter` keeps `notify` and `.model`; it now delegates the state to a
`CircuitBreaker` instance. `ASRRouter` does the same for `engine_name`/`notify`.
Both routers pass `clock` through for deterministic tests.

## 6. Router state machine (`asr_engine: "qwen3"` == auto)

Mirrors §5 of the LLM spec, with `transcribe` as the single dispatched method.

- **`local` (SenseVoice-pinned):** `set_engine("sensevoice")` → every `transcribe`
  goes straight to SenseVoice; the remote is never contacted. `set_engine` calls
  `breaker.reset()` so a fresh choice starts clean (mirrors `LLMRouter.set_backend`).
- **`auto` (`set_engine("qwen3")`):**
  - breaker healthy → remote `transcribe`.
    - success → `record_success()`; if it had been tripped, notify `("reconnected",
      "qwen3")`; `engine_name = "qwen3"`.
    - `ASRUnavailable` → `record_failure()`; serve this utterance from SenseVoice;
      `engine_name = "sensevoice"`.
  - **Trip:** 3rd consecutive failure → breaker temporary; notify `("fallback",
    "sensevoice")`; while cooling, `transcribe` goes **straight to SenseVoice** (no
    remote attempt, no ~1 s stall per utterance).
  - **Auto-retry:** first call after cooldown re-probes the remote once; success →
    back to remote + "reconnected"; failure → re-arm cooldown, stay local.
- If SenseVoice *itself* raises (model/mic error), that propagates unchanged — the
  existing `_process_audio` error path handles it (degrade, never crash).

`engine_name` is read by `_process_audio` **after** `transcribe` (app.py:430) so the
log line shows who actually served, including a mid-call fallback — preserved.

## 7. Hotwords — "same set, used two ways"

The call site already passes `context=self._vocab_list()` (=`vocab_terms(dictionary,
hotwords)`) into `transcribe` and already runs `apply_phonetic_hotwords` post-hoc for
every engine. So:
- **Remote Qwen3-ASR:** `context` → request `prompt` → native decode-time biasing.
- **SenseVoice fallback:** `context` ignored by the model (no biasing), but the
  downstream `apply_phonetic_hotwords` recovers homophone slips deterministically —
  exactly today's behavior.

No new config, no second list, no pipeline change. The hotword question reduces to
"feed the same `vocab_terms` list into the remote request," which §4 does.

## 8. Config additions (`config.py` DEFAULTS)

```python
"asr_engine": "sensevoice",             # unchanged key: "qwen3" (remote+fallback) | "sensevoice"
"qwen_asr_url": "http://127.0.0.1:8001/v1",
"qwen_asr_model": "Qwen/Qwen3-ASR-1.7B",
"qwen_asr_api_key": "",                # optional; environment variable takes precedence
"qwen_asr_connect_timeout": 1.0,        # fast-fail an unreachable box
"qwen_asr_total_timeout": 30.0,         # generous read deadline for one-shot transcription
```
`fallback_threshold` / `fallback_cooldown` are **reused** for both routers (same
breaker policy). The `asr_engine` default stays `sensevoice` so nothing auto-connects
to the remote until the user opts in — matching the user's rule and the existing
default. Timeout/URL/model are advanced knobs (config.json); the menu only toggles
`asr_engine`.

## 9. Menu / UX (`app.py`) — 2-mode engine menu

- The existing **"Recognition Engine"** radio submenu keeps its two codes but is
  relabeled:
  - `engine_qwen3` → **"Qwen3-ASR — remote (auto-fallback)"** → `set_engine("qwen3")`.
  - `engine_sensevoice` → **"SenseVoice — local"** → `set_engine("sensevoice")`.
  - The on-device Qwen3-ASR option is **not** offered (2-mode parity with the LLM).
- Breaker fallback/reconnect surfaces exactly like the LLM's, via a **new**
  `notify` path independent of the LLM's:
  - `_build_asr_router(notify=self._on_asr_switch)`.
  - `_on_asr_switch(event, model)` (worker thread) sets `_asr_switch_note` +
    `_asr_switch_dirty`; `_refresh_ui` consumes it and calls `_apply_asr_switch`.
  - `_apply_asr_switch`: on `"fallback"` tag `menu_engine.title +=
    engine_fallback_tag`, set `status_asr_fallback`, post `notify_asr_fallback_*`;
    on `"reconnected"` clear the tag, set `status_asr_reconnected`, post
    `notify_asr_reconnect_*`. (A breaker trip does **not** rewrite `asr_engine` — the
    user's selection stays "qwen3"; only the title tag + notice change, mirroring the
    LLM's `menu_model` behavior.)
- `_set_engine` also clears the fallback tag when the user picks `sensevoice`
  (mirrors `_set_backend` clearing the LLM tag) and warms SenseVoice off-thread
  (existing `_preload`).

## 10. Error handling & degradation (unchanged contract)

- Remote raises `ASRUnavailable` on its own failures; router serves SenseVoice.
- SenseVoice failure propagates → existing `_process_audio` error path (raw/notice).
- Never blocks the AppKit main thread (transcription already runs on a worker).
- No mocks / stubs / placeholders in product code (SPEC Principle 0 / CLAUDE.md).

## 11. Testing

- **`tests/test_breaker.py`** (new, deterministic, injected clock): trip after
  `threshold`, `allow_remote` gating during cooldown, re-probe after cooldown,
  success resets, `record_success/record_failure` return flags, `reset()`.
- **`tests/test_asr_router.py`** (new; FakeASR backends — real classes with
  in-memory behavior, no product-code mocks; injected clock): remote failure →
  SenseVoice output + counter increments; success resets; 3 failures → tripped, 4th
  call does **not** touch the remote (spy) and serves local; clock past cooldown →
  re-probe; `asr_engine="sensevoice"` → remote never called; notify fires
  `fallback`/`reconnected`; `engine_name` reflects the actual server per call.
- **`tests/test_qwen_asr_backend.py`** (new): parse a `{"text": ...}` body via a
  fake response object (test-side, like `test_vllm_backend`'s `FakeResp`); `prompt`
  and `language` are populated in the multipart fields; connect failure against a
  **real closed port** (`http://127.0.0.1:1/v1`) → `ASRUnavailable`; `ping()` false
  on a closed port.
- **`tests/test_llm_router.py`** stays green unchanged (guards the breaker refactor).
- **`scripts/itest_qwen_asr_live.py`** (new, mirrors `itest_vllm_live.py`): Part A —
  real transcription of `tests/fixtures/yue_en_5s.wav` through `RemoteQwenASRBackend`
  (prints the remote transcript); Part B — forced fallback (dead `qwen_asr_url`) →
  real SenseVoice transcript; prints `RESULT: FALLBACK OK`. If the remote can't
  decode audio yet (§1a), print that explicitly and still demonstrate Part B — never
  fabricate output.
- Full `pytest` stays green; no-mocks grep over `whisperflow_local/` stays clean;
  `--selftest` shows the relabeled 2-item engine submenu.

## 12. Files to change

- `whisperflow_local/asr.py` — add `RemoteQwenASRBackend` + `ASRUnavailable`;
  `SenseVoiceEngine`/`Qwen3ASREngine` unchanged.
- `whisperflow_local/asr_router.py` — **new** `ASRRouter` (fallback + breaker).
- `whisperflow_local/breaker.py` — **new** shared `CircuitBreaker`.
- `whisperflow_local/router.py` — refactor `LLMRouter` to compose `CircuitBreaker`
  (public API unchanged).
- `whisperflow_local/config.py` — new DEFAULTS keys (§8).
- `whisperflow_local/app.py` — build `ASRRouter`; relabel engine menu; ASR
  fallback/reconnect notify path (`_on_asr_switch`/`_apply_asr_switch`/dirty flag in
  `_refresh_ui`); `_set_engine` tag-clear.
- `whisperflow_local/i18n.py` — relabel `engine_qwen3`/`engine_sensevoice`; add
  `engine_fallback_tag`, `status_asr_fallback`, `status_asr_reconnected`,
  `notify_asr_fallback_title/body`, `notify_asr_reconnect_title/body` (en + zh-HK).
- `tests/test_breaker.py`, `tests/test_asr_router.py`,
  `tests/test_qwen_asr_backend.py` — new.
- `scripts/itest_qwen_asr_live.py` — new.
- `SPEC.md` (§1 tech stack) + `CLAUDE.md` — one-line note that the ASR layer is
  remote-primary (vLLM Qwen3-ASR) with local SenseVoice fallback + shared breaker.

## 13. Acceptance criteria

- **AC1** In `auto` with a healthy remote, a real dictation transcribes via the
  remote Qwen3-ASR; the live script prints the remote transcript and `engine_name`
  is `qwen3`. *(Gated on §1a server fix.)*
- **AC2** With the remote unreachable (bad port/host down), a real dictation still
  completes via local SenseVoice within ~1 s of the connect deadline — no
  multi-second hang, no crash; `engine_name` is `sensevoice`.
- **AC3** A long utterance completes on the remote and is **not** cut off early
  (connect fast-fail, generous read deadline). *(Gated on §1a.)*
- **AC4** After 3 consecutive remote failures the app pins to SenseVoice, the engine
  menu shows a fallback tag + a notification; subsequent dictations do not attempt
  the remote (no ~1 s stalls) — verified by the deterministic router test.
- **AC5** After the cooldown, the next dictation re-probes the remote and, if
  healthy, switches back with a notice — verified by the clock-injected test.
- **AC6** Selecting "SenseVoice — local" pins to SenseVoice; the remote is never
  contacted until the user re-selects "Qwen3-ASR — remote (auto-fallback)."
- **AC7** The same `vocab_terms` list biases the remote (via `prompt`) and still
  drives `apply_phonetic_hotwords` for the SenseVoice path; no second list/UI.
- **AC8** `pytest` fully green (incl. unchanged `test_llm_router.py`); no-mocks grep
  over `whisperflow_local/` clean; UI never freezes (ASR off the main thread).

## 14. Execution mode & verification

- **Grounding first.** Read `asr.py`, `router.py`, `app.py` (`_build_menu` engine
  section, `_process_audio`, `_preload`, `_set_engine`, `_build_router`,
  `_on_llm_switch`/`_apply_llm_switch`, `_refresh_ui`, `_sync_checkmarks`),
  `config.py`, `i18n.py`, and `tests/test_llm_router.py`/`test_vllm_backend.py` for
  the patterns to mirror. Run commands; do not infer file contents.
- **Thinking depth.** Think hard about the `CircuitBreaker` extraction (must not
  regress the shipped LLM router) and the `ASRRouter.transcribe` dispatch; routine
  wiring needs no extra reasoning.
- **Test-first** for `breaker.py`, `ASRRouter`, and `RemoteQwenASRBackend` parsing.
- **Completeness (anti-laziness).** Implement the entire §12 scope and AC1–AC8; do
  not deliver ~80% and list the rest as TODO. Stop early only on a genuine blocker.
- **Conventions.** CLAUDE.md: no mocks in product code, never run ASR on the AppKit
  main thread, degrade never crash, all settings via `config.Config`.

Every completion claim needs fresh verification pasted in the transcript:
- `.venv/bin/python -m pytest tests/` → all pass (incl. new + unchanged router test).
- `grep -rniE "mock|fake|dummy|stub|placeholder|hardcode|lorem|TODO" whisperflow_local/`
  → no meaningful hits.
- `.venv/bin/python -m whisperflow_local --selftest` → engine submenu shows the two
  relabeled entries.
- `.venv/bin/python scripts/itest_qwen_asr_live.py` → real remote transcript (AC1,
  once §1a fixed) **and** forced fallback to real SenseVoice (AC2). If the box still
  can't decode audio, print that and still demonstrate the fallback path.
- **Blocker rule.** If the same external blocker (remote can't decode audio, box
  unreachable) recurs 3×, stop autonomous changes and output a blocker report:
  steps tried, evidence, remaining risk, next-step options.
