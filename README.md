# WhisperFlow-Local

Private self-hosted **Cantonese-English dictation** for macOS (Apple Silicon).
Hold a key, speak in 廣東話, English, or 中英夾雜 — the transcript is cleaned up
by models on the user's GB10 and pasted into whatever app you're using. No
third-party cloud service is used.

- **ASR**: `Qwen3-ASR-0.6B` on the private GB10 is primary;
  SenseVoiceSmall remains the on-Mac ASR fallback.
- **AI cleanup**: `Qwen3.6-35B-A3B` through the GB10's OpenAI-compatible vLLM API
  (thinking disabled). Ollama is not used.
- **Everything else**: native menu-bar app (rumps), global hotkeys (pynput),
  clipboard-paste insertion, two formatting modes, content generation, custom dictionary,
  voice commands, SQLite history, launch-at-login.

## Requirements

- macOS on Apple Silicon (built and tested on M4, 16 GB RAM)
- Python 3.11+ (`brew install python@3.11`)
- Tailscale connectivity to the private GB10 (`100.71.138.54`)

## Install & run

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/make_app.sh        # builds ~/Applications/WhisperFlow-Local.app
open ~/Applications/WhisperFlow-Local.app
```

Double-click **WhisperFlow-Local** in `~/Applications` from now on — no
terminal needed. (Launching via the app also means macOS attributes the
Microphone / Accessibility / Input Monitoring permissions to
"WhisperFlow-Local" itself, granted once, regardless of terminal.)

Dev mode alternative: `.venv/bin/python -m whisperflow_local`

**UI language**: menus, dialogs and notifications follow your macOS
language automatically (English / 廣東話 zh-HK); override in
設定 → 介面語言. **Punctuation**: toggle output punctuation on/off in
設定 → 標點符號.

If the remote ASR is unavailable, SenseVoiceSmall (~1 GB) downloads on first
fallback and is cached under `~/.cache/modelscope`.

## Private GB10 models

WhisperFlow is configured for the user's private GB10 by editing:

`~/Library/Application Support/WhisperFlow-Local/config.json`

Only the keys you want to override are required:

```json
{
  "llm_backend": "remote",
  "vllm_url": "http://100.71.138.54:8090/v1",
  "vllm_model": "Qwen3.6-35B-A3B",
  "vllm_api_key": "",
  "asr_engine": "qwen3",
  "qwen_asr_url": "http://100.71.138.54:8800/v1",
  "qwen_asr_model": "Qwen3-ASR-0.6B",
  "qwen_asr_api_key": ""
}
```

Bearer tokens are optional. Prefer the `WHISPERFLOW_VLLM_API_KEY` and
`WHISPERFLOW_QWEN_ASR_API_KEY` environment variables when your launch method
supports them; otherwise the matching config values are stored in a user-only
file (`0600`). Use TLS or a trusted private network for any non-local endpoint.

## Grant permissions (one time)

The app opens these panes for you on first run. In
**System Settings → Privacy & Security**, enable your terminal (or Python)
under:

1. **Microphone** — to record your speech
2. **Accessibility** — to paste text into other apps (synthesized ⌘V)
3. **Input Monitoring** — to detect the global push-to-talk key

Restart the app after granting.

## Use it

| Action | Default |
| --- | --- |
| Push-to-talk | **hold Right Option**, speak, release |
| Hands-free toggle | **⌘⇧D** to start, again to stop |
| Formatting mode | menu → Formatting Mode (原文口語 Verbatim / 書面結構化 Structured) |
| Language | menu → Language (Auto / Cantonese / English / Mixed) |
| Content generation | **hold Shift + Right Option**, speak a request, release |
| History | menu → History (click an entry to re-insert) |

**Voice commands** (spoken, not typed): "new line", "new paragraph" (新段落),
"press enter" / 發送 at the end, "scratch that" / 當我冇講過 to undo the last
dictation, "all caps …".

**Formatting modes**: *Verbatim* keeps your spoken wording and only fixes
punctuation, drops fillers/stutters and corrects ASR homophone slips — a guard
makes it impossible for the model to translate, reorder or summarise you.
*Structured* understands the whole utterance and re-emits it as structured
written Chinese, resolving self-corrections ("坐 89X… 唔係，268") to what you
finally meant.

**Content generation**: hold Shift + Right Option and say what you want written
("draft an email to Lulu saying…"). The result is pasted where you are typing.
If the request is ambiguous the recording pill becomes a small panel asking at
most two questions — click an option or press 1-3, Esc to cancel.

**Custom dictionary**: Settings → Edit Dictionary — fix names and jargon
(e.g. `"kenny" → "Ken Ng"`).

## Manual end-to-end checklist

1. Start the app: `.venv/bin/python -m whisperflow_local` → 🎤 appears in the
   menu bar; wait until the status line says **Ready**.
2. Click into any text field (Notes, a browser, Messages…).
3. **Hold Right Option** and say: 「我聽日要開會，個 project deadline 係 Friday」
4. Release. You hear the stop cue, the icon shows ✍️ then ✨, and the cleaned
   text pastes at your cursor — Cantonese and English both intact.
5. Say "scratch that" to delete it; say "… press enter" to auto-send in a chat app.
6. Stop or disconnect the GB10 and dictate again → deterministic cleanup/raw
   text still pastes; local Ollama is never contacted.

## Verify the pipeline without the mic

```bash
.venv/bin/python scripts/e2e.py   # real TTS audio -> real ASR -> real LLM
.venv/bin/python -m pytest tests/ # unit tests
```

## Optional: build a standalone .app

```bash
scripts/build_app.sh   # produces dist/WhisperFlow-Local.app (unsigned)
```

## Troubleshooting

**Sound cues play, ASR runs, but no text appears** — this is the
Accessibility permission missing. Input Monitoring makes the hotkey work,
but *synthesizing* the ⌘V keystroke needs Accessibility; without it macOS
silently drops the paste. The app now detects this: your text is **copied
to the clipboard** (just press ⌘V) and a notification explains. For
auto-paste: System Settings → Privacy & Security → **Accessibility** →
enable your terminal (the exact app you launch from — Terminal, iTerm,
VS Code…), then **restart the app**. Menu → *Permissions & Setup…* shows a
live ✅/❌ status for each permission.

Every dictation is also written to
`~/Library/Application Support/WhisperFlow-Local/app.log`
(stages: audio → asr → insert) so you can always see what was heard and
where the text went.

**The waveform HUD** — while recording, a native macOS pill (blur/vibrancy,
like the system dictation indicator) floats above the Dock showing your live
waveform; it shimmers while transcribing and fades out when done. Preview it
any time without dictating: `.venv/bin/python scripts/overlay_demo.py`

## Privacy

Audio is sent only to the private GB10 ASR endpoint, and transcripts are sent
only to the private GB10 LLM endpoint over Tailscale. Local Ollama and
third-party cloud APIs are not used; no telemetry is included. History lives in
`~/Library/Application Support/WhisperFlow-Local/history.sqlite3` — clear it any
time from the menu.
