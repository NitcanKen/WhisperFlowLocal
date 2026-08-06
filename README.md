# WhisperFlow-Local

Local-first **Cantonese-English dictation** for macOS (Apple Silicon).
Hold a key, speak in 廣東話, English, or 中英夾雜 — the transcript is cleaned up
by a local LLM and pasted into whatever app you're using. No third-party cloud
service is required; self-hosted remote models are available as an explicit opt-in.

- **ASR**: [SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
  via funasr — the best-benchmarked open model for Cantonese-English
  code-switching (~9% CER on mixed speech).
- **AI cleanup**: `qwen3.5:4b` via your local [Ollama](https://ollama.com)
  (thinking mode disabled for instant, clean output). Optional — toggle it off
  for raw transcripts.
- **Everything else**: native menu-bar app (rumps), global hotkeys (pynput),
  clipboard-paste insertion, per-app formatting profiles, custom dictionary,
  voice commands, SQLite history, launch-at-login.

## Requirements

- macOS on Apple Silicon (built and tested on M4, 16 GB RAM)
- Python 3.11+ (`brew install python@3.11`)
- [Ollama](https://ollama.com) running locally with the model pulled:

```bash
ollama list            # confirm qwen3.5:4b is present
ollama pull qwen3.5:4b # only if missing
```

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

On first run the SenseVoiceSmall model (~1 GB) downloads automatically and is
cached under `~/.cache/modelscope`; afterwards the app is fully offline
(the only network traffic is to `localhost` Ollama).

## Optional self-hosted remote models

WhisperFlow starts in local-only mode. Advanced users can opt into an
OpenAI-compatible vLLM server for LLM cleanup, Qwen3-ASR, or both by editing:

`~/Library/Application Support/WhisperFlow-Local/config.json`

Only the keys you want to override are required:

```json
{
  "llm_backend": "auto",
  "vllm_url": "http://vllm-host.example:8000/v1",
  "vllm_api_key": "",
  "asr_engine": "qwen3",
  "qwen_asr_url": "http://asr-host.example:8001/v1",
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
| Formatting profile | menu → Formatting Profile (Raw / Clean / Email / Message / Notes) |
| Language | menu → Language (Auto / Cantonese / English / Mixed) |
| AI commands | menu → AI Commands (Formalize / Summarize / Translate ⇄) |
| History | menu → History (click an entry to re-insert) |

**Voice commands** (spoken, not typed): "new line", "new paragraph" (新段落),
"press enter" / 發送 at the end, "scratch that" / 當我冇講過 to undo the last
dictation, "all caps …".

**Per-app profiles**: Mail auto-formats as Email, Messages/Slack as chat,
Terminal/VS Code stays Raw — editable in Settings → Edit Per-App Rules.

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
6. Quit Ollama and dictate again → the raw transcript still pastes (graceful
   degradation), with a notice in the menu.

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

By default, audio and dictated text never leave your Mac. Network access is
limited to the first-run model download and local Ollama. If you explicitly
enable a remote ASR backend, audio is sent to the configured endpoint; if you
enable a remote LLM backend, transcripts are sent to that endpoint. No
third-party telemetry is included. History lives in
`~/Library/Application Support/WhisperFlow-Local/history.sqlite3` — clear it any
time from the menu.
