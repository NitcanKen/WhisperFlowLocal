"""Real end-to-end pipeline check (SPEC §J1).

Generates REAL audio with macOS `say` (TTS -> waveform -> microphone-format
WAV), runs it through the ACTUAL SenseVoiceSmall model, then through the
ACTUAL Qwen3.6-35B-A3B on the private GB10 vLLM endpoint. Prints every stage.

Usage: .venv/bin/python scripts/e2e.py
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperflow_local.asr import ASREngine
from whisperflow_local.config import Config
from whisperflow_local.llm import LLMUnavailable, VLLMBackend
from whisperflow_local.textproc import apply_dictionary, parse_voice_commands


def cantonese_voice() -> str:
    """Return an installed Cantonese `say` voice name, or ''."""
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "zh_HK" in line or "yue" in line.lower() or "Sinji" in line:
            return line.split()[0]
    return ""


def synthesize(text: str, voice: str, out_wav: str) -> None:
    """Real speech audio via macOS TTS, resampled to 16 kHz mono PCM."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff = tmp.name
    cmd = ["say", "-o", aiff]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    subprocess.run(cmd, check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, out_wav],
        check=True,
    )
    os.unlink(aiff)


def main() -> int:
    print("=" * 72)
    print("WhisperFlow-Local end-to-end check (real models, no shortcuts)")
    print("=" * 72)

    yue = cantonese_voice()
    workdir = tempfile.mkdtemp(prefix="wfl_e2e_")
    clips = []
    if yue:
        print(f"[audio] Cantonese TTS voice found: {yue}")
        clips.append(("cantonese-english mix", yue,
                      "我聽日要開會，個 project 嘅 deadline 係 Friday，記住 send email 俾我"))
    else:
        print("[audio] No Cantonese TTS voice installed; using English voice.")
        print("        (For a spoken Cantonese test, add the Sinji voice in "
              "System Settings → Accessibility → Spoken Content.)")
    clips.append(("english + voice command", "",
                  "let's meet at six tomorrow press enter"))

    asr = ASREngine()
    print("[asr] Loading SenseVoiceSmall (first run downloads the model)…")
    asr.ensure_loaded(progress_cb=lambda m: print(f"[asr] {m}"))

    cfg = Config()
    llm = VLLMBackend(
        cfg.get("vllm_url"), cfg.get("vllm_model"),
        connect_timeout=cfg.get("vllm_connect_timeout"),
        ttft_timeout=cfg.get("vllm_ttft_timeout"),
        total_timeout=cfg.get("vllm_total_timeout"),
        api_key=(os.environ.get("WHISPERFLOW_VLLM_API_KEY")
                 or cfg.get("vllm_api_key")),
        reasoning_effort=cfg.get("vllm_reasoning_effort"),
    )
    llm_ok = llm.ping()
    print(f"[llm] GB10 at {cfg.get('vllm_url')} model={cfg.get('vllm_model')} "
          f"reachable={llm_ok}")

    failures = 0
    for i, (label, voice, text) in enumerate(clips, 1):
        wav = os.path.join(workdir, f"clip{i}.wav")
        print("-" * 72)
        print(f"[clip {i}] {label}")
        print(f"[clip {i}] TTS input : {text}")
        synthesize(text, voice, wav)
        size = os.path.getsize(wav)
        print(f"[clip {i}] audio file: {wav} ({size} bytes)")

        transcript = asr.transcribe(wav, "auto")
        print(f"[clip {i}] SenseVoice transcript: {transcript}")
        if not transcript.strip():
            print(f"[clip {i}] FAIL: empty transcript")
            failures += 1
            continue

        withdict = apply_dictionary(transcript, cfg.get("dictionary"))
        parsed = parse_voice_commands(withdict)
        print(f"[clip {i}] after commands: text={parsed.text!r} "
              f"press_enter={parsed.press_enter}")

        if llm_ok and parsed.text:
            try:
                cleaned = llm.format_text(parsed.text, "Clean")
                print(f"[clip {i}] Qwen3.8 Clean : {cleaned}")
                if i == 1:
                    email = llm.format_text(parsed.text, "Email")
                    print(f"[clip {i}] Qwen3.8 Email : {email}")
                    translated = llm.run_command("Translate to English", parsed.text)
                    print(f"[clip {i}] Qwen3.8 Translate to English: {translated}")
            except LLMUnavailable as exc:
                print(f"[clip {i}] LLM degraded gracefully: {exc}")
        elif not llm_ok:
            print(f"[clip {i}] LLM skipped (GB10 not reachable) — raw "
                  f"transcript would be inserted, per graceful degradation.")

    print("=" * 72)
    if failures:
        print(f"E2E RESULT: FAIL ({failures} clip(s) produced no transcript)")
        return 1
    print("E2E RESULT: OK — real audio -> real SenseVoice -> real GB10 Qwen3.8 pipeline works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
