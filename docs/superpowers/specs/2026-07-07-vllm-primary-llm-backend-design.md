# Remote vLLM primary LLM backend with local Ollama fallback — Design

Date: 2026-07-07
Status: Approved (design), ready for implementation

## 1. Context & motivation

Today the LLM cleanup layer (`whisperflow_local/llm.py`) talks only to the local
Ollama `/api/chat` API with `qwen3.5:4b`. The user now runs a much stronger model
on a separate machine: **`nvidia/Qwen3.6-35B-A3B-NVFP4` served by vLLM** over the
**OpenAI-compatible API** at `http://redacted-host:8000/v1` (reachable via
Tailscale). Confirmed live: `GET /v1/models` returns that model id,
`max_model_len` 262144.

Goal: make the **remote 35B the primary** LLM for all cleanup / formatting / AI
commands, with the **local `qwen3.5:4b` Ollama as an automatic fallback** when the
remote is slow or unreachable, plus a **circuit breaker** so a persistently-down
remote stops costing latency on every dictation.

## 2. Goals / non-goals

**Goals**
- Remote 35B (OpenAI `/v1`) is primary in `auto` mode; local Ollama is fallback.
- Per-request fallback on remote trouble, using a **time-to-first-token (TTFT)
  deadline** (not a total-time deadline) so long legitimate generations still use
  the 35B while a dead/asleep/overloaded remote fails over within ~1s.
- Circuit breaker: after **3 consecutive fallbacks**, switch to local-only; after
  a **cooldown (default 5 min)** auto-retry the remote and switch back on success.
- Menu lets the user pin **local-only** (`qwen3.5:4b`), which never touches vLLM.
- Preserve every existing behavior: thinking disabled, guarded edit-list for
  Clean, vocab injection, jyutping hint, graceful degradation, no UI freeze.

**Non-goals**
- No streaming of partial text into the target app (we still paste once, whole).
  Streaming is used only internally to measure TTFT.
- No change to ASR, hotkeys, injection, history, or the Ollama wire format.
- No auth on the vLLM endpoint (LAN/Tailscale only).

## 3. Architecture (Approach A: base + two backends + router)

Refactor `llm.py` so the **prompt-building logic is shared** and only the **wire
format differs** per backend, then put fallback/breaker logic in a separate router.

```
BaseLLMBackend                     # shared: PROFILES, EDIT_SYSTEM, AI_COMMANDS,
  .format_text(text, profile, vocab)   #   vocab injection, jyutping hint,
  .propose_edits(text, vocab)          #   parse_edits, <think> strip.
  .run_command(command, text)          # each builds prompts, calls self._chat(...)
  .ping()                              # abstract per backend
  ._chat(system, user, force_json)   # ABSTRACT

OllamaBackend(BaseLLMBackend)        # today's code: POST /api/chat, think:false,
                                     #   format:json, non-streaming. ping=/api/version
VLLMBackend(BaseLLMBackend)          # POST /v1/chat/completions, stream:true + TTFT,
                                     #   response_format json_object, enable_thinking:false.
                                     #   ping=/v1/models

LLMRouter                            # same 4 public methods; owns fallback + breaker.
  .format_text / .propose_edits / .run_command / .ping   # delegate via _dispatch(name, *args)
  .set_backend(mode) / .set_local_model(m) / .set_remote(url, model)
```

**Backward compatibility:** keep `LLMClient = OllamaBackend` as an alias. Existing
tests (`test_vocab` stubs `_chat` at the client boundary; `test_llm_degradation`
uses a real closed socket) continue to pass unchanged.

`app.py` builds `self.llm = LLMRouter(...)` instead of `LLMClient(...)`. All call
sites (`_process_audio` → `propose_edits`/`format_text`, `_ai_command` →
`run_command`) are **unchanged** because the router exposes the same interface and
raises the same `LLMUnavailable` contract when *both* backends are unavailable.

## 4. TTFT streaming design (the core of the timeout behavior)

A hard 1s *total* timeout is rejected: the 35B cannot finish Email/Notes/Summarize
in 1s, so it would never be used. A generous total timeout is also wrong: an
asleep/queued remote would hang for seconds. Solution — **stream and gate on
first-token latency**:

`VLLMBackend._chat` issues `POST /v1/chat/completions` with `stream: true` and:
- `timeout=(vllm_connect_timeout, vllm_ttft_timeout)` on `requests.post(...,
  stream=True)`. The connect timeout (~1s) catches unreachable/asleep hosts; the
  **read timeout (~1s) is the TTFT deadline** — if the server sends no first byte
  within ~1s (down / crashed / queued behind other work), `requests` raises
  `ReadTimeout`. It also doubles as a stalled-stream detector between chunks
  (real vLLM token gaps are tens of ms, far under 1s).
- A wall-clock **total safety cap** (`vllm_total_timeout`, default 30s) checked in
  the accumulation loop, to abort a stream that dribbles forever.
- Accumulate `choices[0].delta.content` from the SSE lines until `[DONE]`;
  defensively strip `<think>…</think>`; return the text.

Any of {connect error, read/TTFT timeout, HTTP != 2xx, malformed stream,
total-cap exceeded} → raise **`LLMUnavailable`** (same contract Ollama uses). The
router treats that as "remote failed, fall back."

Rationale: a healthy 35B over Tailscale returns its first token in tens of ms, so
~1s TTFT is generous for "healthy" yet snappy for "dead," and long generations are
never cut off once they've started.

## 5. Router state machine (`llm_backend: "auto"`)

State in `LLMRouter`, guarded by a `threading.Lock` (dictations run on worker
threads). An **injectable clock** (`clock=time.monotonic`) makes the breaker
deterministically testable.

- **`local` (user-pinned):** every call goes straight to Ollama; vLLM never
  called. Set when the user picks "qwen3.5:4b — local" in the menu.
- **`auto` mode has two internal states:**
  - **REMOTE (normal):** call vLLM (streaming + TTFT).
    - success → `consecutive_fallbacks = 0`; if we were TEMPORARY, log/notify
      "reconnected to remote" and go REMOTE.
    - `LLMUnavailable` → `consecutive_fallbacks += 1`; serve this request from the
      local Ollama backend and return its result.
  - **Trip:** when `consecutive_fallbacks >= fallback_threshold` (3) → enter
    **LOCAL_TEMPORARY**, set `retry_at = clock() + fallback_cooldown` (300s),
    notify the user (menu radio flips to local + a notification). While TEMPORARY,
    calls go **straight to Ollama** — no vLLM attempt, no 1s stalls.
  - **Auto-retry:** the first call after `clock() >= retry_at` attempts vLLM once
    (on its worker thread). Success → REMOTE + reset counter + "reconnected"
    notice. Failure → re-stamp `retry_at`, stay TEMPORARY.

If Ollama *also* raises `LLMUnavailable` (both down), propagate it — `app.py`
already degrades to pasting raw ASR with a notice.

`_dispatch(name, *args, **kwargs)` centralizes this: `format_text` /
`propose_edits` / `run_command` / `ping` each just call
`self._dispatch("<name>", ...)`, so the fallback logic exists once.

## 6. vLLM wire-format specifics

- Endpoint: `POST {vllm_url}/chat/completions` (vllm_url already ends in `/v1`).
- Body: `model`, `messages` (same system/user as Ollama), `stream: true`,
  top-level `temperature` (0 for `force_json`, else 0.2).
- JSON mode (edit list): `response_format: {"type": "json_object"}`.
- **Thinking disabled:** `chat_template_kwargs: {"enable_thinking": false}` (vLLM
  extra body for Qwen3 chat template) **and** the existing `<think>…</think>`
  strip as belt-and-suspenders. (No `think` field — that is Ollama-only.)
- Health probe (`ping`): `GET {vllm_url}/models`, short timeout.

## 7. Config additions (`config.py` DEFAULTS)

```python
"llm_backend": "auto",                 # "auto" (remote primary + fallback) | "local"
"vllm_url": "http://redacted-host:8000/v1",
"vllm_model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
"vllm_connect_timeout": 1.0,
"vllm_ttft_timeout": 1.0,              # seconds to first token before fallback
"vllm_total_timeout": 30.0,           # wall-clock safety cap once streaming
"fallback_threshold": 3,              # consecutive fallbacks before pinning local
"fallback_cooldown": 300.0,           # seconds before auto-retrying the remote
```
Existing `ollama_url` / `ollama_model` / `llm_enabled` unchanged (local backend).
Backend/timeout/threshold values are advanced knobs (editable in config.json); the
menu only toggles `llm_backend`.

## 8. Menu / UX (`app.py`)

- New **"AI Model"** radio submenu (built next to the ASR "Engine" submenu):
  - **"Qwen3.6-35B — remote (auto-fallback)"** → `config.set("llm_backend","auto")`
    + `self.llm.set_backend("auto")`.
  - **"qwen3.5:4b — local"** → `config.set("llm_backend","local")` +
    `self.llm.set_backend("local")`.
  - Radio checkmark reflects the live effective backend; the breaker flipping to
    TEMPORARY updates the checkmark to local and posts a notification, and back on
    reconnect. Menu updates marshal to the main thread the same way the existing
    Ollama-down notice does (reuse that mechanism).
- Existing Settings → "Set Model…" dialog stays and edits the **local** ollama
  model string, calling `self.llm.set_local_model(...)`.

## 9. Error handling & degradation (unchanged contract)

- Both backends raise `LLMUnavailable` on their own failures.
- Router: remote failure → local; local failure (or local-only + local down) →
  `LLMUnavailable` propagates → `app.py` pastes raw ASR + notice (today's path).
- Never blocks the AppKit main thread (all calls already on worker threads).
- No mocks / stubs / placeholders in product code (SPEC Principle 0).

## 10. Testing

- **`tests/test_llm_router.py`** (deterministic, injected clock; backends stubbed
  at the `_chat` boundary exactly like `test_vocab` — no product-code mocks):
  - Remote failure → router returns the local backend's output; counter increments.
  - Remote success → counter resets to 0.
  - 3 consecutive failures → mode TEMPORARY; the 4th call does **not** invoke the
    remote (spy) and is served locally.
  - Advancing the injected clock past `fallback_cooldown` → next call re-attempts
    remote; success → REMOTE; failure → stays TEMPORARY, `retry_at` re-stamped.
  - `llm_backend="local"` → remote never invoked regardless of breaker.
- **`tests/test_vllm_backend.py`**: `parse`/accumulate an SSE stream (fake line
  iterator — test-side, not product), `<think>` stripping, and connect failure
  against a **real closed port** → `LLMUnavailable` (mirrors `test_llm_degradation`).
- **`scripts/itest_vllm_live.py`**: real 35B round-trip through `VLLMBackend`
  (Clean edit list + one profile), printed output; then a bad-`vllm_url` run
  showing real fallback to the real local Ollama (mirrors `itest_postcorrect_live`).
- Full `pytest` stays green; no-mocks grep over `whisperflow_local/` stays clean.

## 11. Files to change

- `whisperflow_local/llm.py` — extract `BaseLLMBackend`; `OllamaBackend`
  (= today's `_chat`, `LLMClient` alias); add `VLLMBackend` (streaming/TTFT).
- `whisperflow_local/router.py` — **new** `LLMRouter` (fallback + breaker).
- `whisperflow_local/config.py` — new DEFAULTS keys (§7).
- `whisperflow_local/app.py` — build `LLMRouter`; "AI Model" radio submenu;
  wire breaker notices + menu-state updates; `set_model` → `set_local_model`.
- `tests/test_llm_router.py`, `tests/test_vllm_backend.py` — new.
- `scripts/itest_vllm_live.py` — new.
- `SPEC.md` (§1 tech stack, §C LLM) + `CLAUDE.md` — one-line note that the LLM
  layer is remote-primary (vLLM/OpenAI) with local Ollama fallback + breaker.

## 12. Acceptance criteria

- **AC1** In `auto`, a healthy remote handles a real dictation via the 35B; output
  printed by the live script is the 35B's (not the 4B's).
- **AC2** With the remote unreachable (bad port / host down), a real dictation
  still completes via the local 4B within ~1s of the TTFT deadline — no multi-second
  hang, no crash.
- **AC3** A long-output profile (Email/Notes) or AI command completes on the 35B
  and is **not** cut off at 1s (TTFT gate, not total).
- **AC4** After 3 consecutive remote failures the app switches to local-only, the
  menu shows local + a notification; subsequent dictations do not attempt the
  remote (no 1s stalls) — verified by the deterministic router test.
- **AC5** After the cooldown, the next dictation re-probes the remote and, if
  healthy, switches back to the 35B with a notice — verified by the clock-injected
  test.
- **AC6** Selecting "qwen3.5:4b — local" pins to Ollama; the remote is never
  contacted until the user switches back to "remote (auto-fallback)."
- **AC7** Thinking stays disabled on the 35B (no `<think>` blocks in output);
  Clean still uses the guarded edit-list; vocab + jyutping hint still applied.
- **AC8** `pytest` fully green; no-mocks grep over `whisperflow_local/` clean;
  UI never freezes (all LLM work off the main thread).

## 13. Execution mode (for the autonomous run)

- **Grounding first.** Before editing, actually read `whisperflow_local/llm.py`,
  `app.py` (`_build_menu`, `_process_audio`, `_ai_command`, `_set_model`, and the
  LLM construction near line 70), `config.py`, and the existing tests
  (`tests/test_vocab.py`, `tests/test_llm_degradation.py`, `tests/conftest.py`).
  Run commands; do not infer file contents.
- **Thinking depth.** Think hard before implementing `VLLMBackend` streaming/TTFT
  and the `LLMRouter` breaker — those are the non-trivial parts. Routine wiring
  needs no extra reasoning.
- **No plan mode.** This design is already approved; implement directly. (Plan
  mode would pause the `/goal` loop waiting for approval.)
- **Subagents.** The core (`llm.py` → `router.py` → `app.py`) is interdependent —
  implement it directly and sequentially with shared context. Fan out subagents in
  a single turn only for genuinely independent artifacts (e.g. authoring the two
  new test files or the live script) after the interfaces are fixed. Do not
  over-spawn.
- **Completeness (anti-laziness).** Implement the entire §11 scope and all AC1–AC8.
  Do not finish ~80% and list the remainder as TODO/gaps as if that were delivery.
  Stop early only on a genuine blocker.
- **Conventions.** Follow `CLAUDE.md`: no mocks in product code, thinking off,
  guarded edit-list for Clean, all LLM work off the AppKit main thread.

## 14. Verification & stopping rules

Every completion claim needs fresh verification surfaced in the transcript. Run
and paste:
- `.venv/bin/python -m pytest tests/` → all pass (exit 0), including new
  `test_llm_router.py` and `test_vllm_backend.py`.
- `grep -rniE "mock|fake|dummy|stub|placeholder|hardcode|lorem|TODO" whisperflow_local/`
  → no meaningful hits.
- `.venv/bin/python -m whisperflow_local --selftest` → menu tree includes the
  "AI Model" submenu (Remote 35B / Local 4B).
- `.venv/bin/python scripts/itest_vllm_live.py` → real 35B output via
  `VLLMBackend` (AC1/AC3) **and** a bad-`vllm_url` run showing real fallback to the
  local 4B (AC2). If the remote is unreachable in this environment, print that
  explicitly and still demonstrate the fallback path — never fabricate output.
- Any failing build/test/grep/selftest ⇒ do not claim completion; list the failing
  command, error summary, what was fixed, and what remains.
- **Blocker rule.** If the same external blocker (remote unreachable, missing
  Ollama model, dependency) recurs 3×, stop autonomous changes and output a
  blocker report: steps tried, evidence, remaining risk, next-step options.
