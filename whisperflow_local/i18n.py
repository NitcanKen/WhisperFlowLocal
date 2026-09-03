"""UI localization: English + zh-HK (香港繁體/廣東話風格).

`ui_language` config: "auto" (follow macOS preferred language), "en", "zh-HK".
Internal identifiers (profile keys, AI command keys, config values) stay
English; only displayed text is translated.
"""

# key: (English, zh-HK)
S = {
    "status_starting": ("Starting…", "啟動中…"),
    "status_loading_model": (
        "Loading SenseVoiceSmall (downloads on first run)…",
        "載入緊 SenseVoiceSmall（首次會自動下載）…",
    ),
    "status_ready": (
        "Ready — hold your push-to-talk key and speak.",
        "就緒 — 按住講話鍵開始口述。",
    ),
    "status_recording": (
        "Recording… release key / toggle to stop",
        "錄音中… 放開按鍵或再按快捷鍵停止",
    ),
    "status_transcribing": ("Transcribing…", "轉寫中…"),
    "status_recording_gen": (
        "Recording a request… release to generate",
        "錄緊你嘅要求… 放開就開始生成",
    ),
    "status_generating": ("Writing…", "生成緊…"),
    "status_clarifying": ("Need one thing first…", "想問你一樣嘢先…"),
    "status_gen_cancelled": ("Generation cancelled.", "已取消生成。"),
    "clarify_hint": (
        "press 1-3 to choose  ·  esc to write it now",
        "撳 1-3 揀  ·  esc 唔使問，直接寫",
    ),
    "clarify_other": ("or type your own answer…", "或者喺度打你嘅答案…"),
    "status_gen_empty": (
        "The model returned nothing — nothing inserted.",
        "模型冇返任何內容 — 冇插入任何嘢。",
    ),
    "status_gen_failed": (
        "Generation failed ({err}) — nothing inserted.",
        "生成失敗（{err}）— 冇插入任何嘢。",
    ),
    "status_too_short": ("Too short — ignored.", "太短 — 已略過。"),
    "status_heard_nothing": ("Heard nothing recognizable.", "聽唔到可辨識嘅內容。"),
    "status_scratched": ("Scratched last dictation.", "已刪除上一段口述。"),
    "status_nothing_to_scratch": ("Nothing to scratch.", "冇嘢可以刪除。"),
    "status_reinserted": ("Re-inserted from history.", "已由歷史記錄重新插入。"),
    "status_copied_last": ("Copied last transcription.", "已複製最近一條轉寫。"),
    "status_copied_no_perm": (
        "⚠️ Copied (⌘V to paste) — grant Accessibility for auto-paste",
        "⚠️ 已複製（按 ⌘V 貼上）— 要自動貼上請授權「輔助使用」",
    ),
    "status_inserted_via": ("Inserted", "已插入"),
    "status_formatting": ("Formatting", "格式化中"),
    "status_llm_off_raw": (
        "LLM off ({err}) — inserted raw transcript.",
        "LLM 未能使用（{err}）— 已插入原始轉寫。",
    ),
    "menu_toggle": ("Start/Stop Recording (hands-free)", "開始／停止錄音（免持）"),
    "menu_profile": ("Formatting Mode", "格式化模式"),
    "menu_language": ("ASR Language", "辨識語言"),
    "menu_engine": ("ASR Engine", "辨識引擎"),
    "engine_sensevoice": ("SenseVoice — local", "SenseVoice — 本地"),
    "engine_qwen3": (
        "Qwen3-ASR — remote (auto-fallback)",
        "Qwen3-ASR — 遠端（自動後備）",
    ),
    "engine_fallback_tag": (" — ⚠︎ SenseVoice fallback", " — ⚠︎ 改用 SenseVoice"),
    "status_asr_fallback": (
        "Remote Qwen3-ASR unreachable — using local SenseVoice.",
        "遠端 Qwen3-ASR 未能連接 — 改用本地 SenseVoice。",
    ),
    "status_asr_reconnected": (
        "Reconnected to remote Qwen3-ASR.",
        "已重新連接遠端 Qwen3-ASR。",
    ),
    "notify_asr_fallback_title": (
        "Switched to local ASR",
        "已切換到本地辨識",
    ),
    "notify_asr_fallback_body": (
        "Remote Qwen3-ASR unreachable after repeated tries — using local "
        "SenseVoice. Will retry the remote automatically.",
        "遠端 Qwen3-ASR 多次連接失敗 — 暫用本地 SenseVoice，稍後會自動重試遠端。",
    ),
    "notify_asr_reconnect_title": (
        "Back on remote ASR",
        "已切換回遠端辨識",
    ),
    "notify_asr_reconnect_body": (
        "Remote Qwen3-ASR reachable again.",
        "遠端 Qwen3-ASR 恢復連接。",
    ),
    "menu_llm": ("AI Cleanup (GB10)", "AI 清理（GB10）"),
    "menu_model": ("AI Model", "AI 模型"),
    "model_remote": (
        "Qwen3.6-35B-A3B — GB10 only",
        "Qwen3.6-35B-A3B — 只用 GB10",
    ),
    "model_auto": (
        "Qwen3.6-35B — remote (auto-fallback)",
        "Qwen3.6-35B — 遠端（自動後備）",
    ),
    "model_local": ("qwen3.5:4b — local", "qwen3.5:4b — 本地"),
    "model_fallback_tag": (" — ⚠︎ local fallback", " — ⚠︎ 本地後備"),
    "status_llm_fallback": (
        "Remote LLM unreachable — using local {model}.",
        "遠端 LLM 未能連接 — 改用本地 {model}。",
    ),
    "status_llm_reconnected": (
        "Reconnected to remote LLM ({model}).",
        "已重新連接遠端 LLM（{model}）。",
    ),
    "notify_llm_fallback_title": (
        "Switched to local AI model",
        "已切換到本地 AI 模型",
    ),
    "notify_llm_fallback_body": (
        "Remote unreachable after repeated tries — using local {model}. "
        "Will retry the remote automatically.",
        "遠端多次連接失敗 — 暫用本地 {model}，稍後會自動重試遠端。",
    ),
    "notify_llm_reconnect_title": (
        "Back on remote AI model",
        "已切換回遠端 AI 模型",
    ),
    "notify_llm_reconnect_body": (
        "Remote LLM reachable again — using {model}.",
        "遠端 LLM 恢復連接 — 使用 {model}。",
    ),
    "menu_history": ("History", "歷史記錄"),
    "menu_history_empty": ("(no transcriptions yet)", "（未有轉寫記錄）"),
    "menu_copy_last": ("Copy Last", "複製最近一條"),
    "menu_clear_history": ("Clear History", "清除歷史記錄"),
    "menu_settings": ("Settings", "設定"),
    "menu_set_ptt": ("Set Push-to-Talk Key…", "設定按住講話鍵…"),
    "menu_set_toggle": ("Set Hands-free Hotkey…", "設定免持快捷鍵…"),
    "menu_vocab": ("Vocabulary (Hotwords)", "詞彙（Hotwords）"),
    "vocab_empty": ("(no terms yet — add one below)", "（未有詞彙 — 喺下面新增）"),
    "vocab_add": ("+ Add Term…", "+ 新增詞彙…"),
    "dlg_vocab_prompt": (
        "Add a term (e.g. WhisperFlow), or a replacement like:  維斯帕 → WhisperFlow",
        "輸入一個詞（例如 WhisperFlow），或者替換規則：  維斯帕 → WhisperFlow",
    ),
    "dlg_vocab_invalid": (
        "Couldn't parse that — use a bare term, or 「wrong → right」.",
        "睇唔明個格式 — 請輸入單一個詞，或者「錯字 → 正字」。",
    ),
    "dlg_vocab_delete": ("Delete 「{item}」?", "刪除「{item}」？"),
    "menu_copy_only": ("Copy Only (no auto-paste)", "只複製（唔自動貼上）"),
    "menu_punct": ("Punctuation", "標點符號"),
    "menu_sounds": ("Sound Cues", "音效提示"),
    "menu_login": ("Launch at Login", "登入時自動啟動"),
    "menu_open_config": ("Open Config Folder", "打開設定資料夾"),
    "menu_ui_lang": ("UI Language (界面語言)", "介面語言 (UI Language)"),
    "menu_permissions": ("Permissions & Setup…", "權限與設定…"),
    "menu_quit": ("Quit", "結束"),
    "profile_Verbatim": ("Verbatim", "原文口語"),
    "profile_Structured": ("Structured", "書面結構化"),
    "lang_auto": ("Auto", "自動"),
    "lang_yue": ("Cantonese 廣東話", "廣東話"),
    "lang_en": ("English", "英文"),
    "lang_mixed": ("Mixed 中英夾雜", "中英夾雜"),
    "cap_title_ptt": ("Set Push-to-Talk Key", "設定按住講話鍵"),
    "cap_title_toggle": ("Set Hands-free Hotkey", "設定免持快捷鍵"),
    "cap_prompt": (
        "Press the key you want… (esc to cancel)",
        "請按你想用嘅按鍵…（esc 取消）",
    ),
    "cap_prompt_combo": (
        "Hold modifiers (⌘⇧⌥⌃), then press a key… (esc to cancel)",
        "㩒住修飾鍵（⌘⇧⌥⌃），再按一個鍵…（esc 取消）",
    ),
    "cap_saved": ("Saved: {key}", "已儲存：{key}"),
    "dlg_model_prompt": ("Ollama model tag for the AI layer:", "AI 層使用嘅 Ollama 模型："),
    "dlg_invalid_json_rules": (
        "Invalid JSON — rules unchanged.",
        "JSON 格式錯誤 — 規則未有更改。",
    ),
    "dlg_restart_lang": (
        "UI language saved. Quit and reopen the app to apply.",
        "介面語言已儲存。請結束並重開 app 以套用。",
    ),
    "notify_perm_title": (
        "Accessibility permission needed",
        "需要「輔助使用」權限",
    ),
    "notify_perm_body": (
        "Text will be COPIED to the clipboard (press ⌘V yourself) "
        "until you grant Accessibility and restart.",
        "授權「輔助使用」並重開 app 之前，文字只會複製到剪貼簿（請自行按 ⌘V 貼上）。",
    ),
    "notify_copied_title": (
        "Copied to clipboard — press ⌘V",
        "已複製到剪貼簿 — 請按 ⌘V",
    ),
    "notify_copied_body": (
        "Auto-paste needs the Accessibility permission. "
        "Menu → Permissions & Setup to grant it, then restart.",
        "自動貼上需要「輔助使用」權限。喺選單 → 權限與設定 授權後重開 app。",
    ),
    "notify_llm_unavailable_title": (
        "GB10 AI unavailable",
        "GB10 AI 未能連接",
    ),
    "notify_mic_title": ("Microphone error", "麥克風錯誤"),
    "notify_mic_body": (
        "Check System Settings → Privacy & Security → Microphone.",
        "請檢查 系統設定 → 隱私權與安全性 → 麥克風。",
    ),
    "notify_model_fail": ("Model load failed", "模型載入失敗"),
    "notify_pipeline_err": ("Pipeline error", "處理過程出錯"),
    "onboard_title": ("{app} — Permissions Status", "{app} — 權限狀態"),
    "onboard_body": (
        "\nHold {ptt} to dictate; {toggle} toggles hands-free mode.\n"
        "Restart the app after granting a permission.\n\nOpening the privacy panes now…",
        "\n按住 {ptt} 口述；{toggle} 切換免持模式。\n"
        "授權任何權限之後請重開 app。\n\n而家幫你打開私隱設定面板…",
    ),
    "perm_summary": (
        "{im} Input Monitoring — global push-to-talk key\n"
        "{ax} Accessibility — auto-paste (⌘V) into other apps\n"
        "🎙 Microphone — macOS asks on first recording\n\n",
        "{im} 輸入監控 — 全域按住講話鍵\n"
        "{ax} 輔助使用 — 自動貼上（⌘V）到其他 App\n"
        "🎙 麥克風 — 首次錄音時 macOS 會自動詢問\n\n",
    ),
    "perm_warning": (
        "⚠️ Without Accessibility, text is COPIED to the clipboard instead "
        "of pasted — press ⌘V yourself, or grant Accessibility to this app "
        "and restart.\n",
        "⚠️ 未授權「輔助使用」時，文字只會複製到剪貼簿（要自行按 ⌘V）。"
        "請喺 隱私權與安全性 → 輔助使用 加入本 App 之後重開。\n",
    ),
}

_ZH = False


def system_prefers_chinese() -> bool:
    try:
        from Foundation import NSLocale

        langs = NSLocale.preferredLanguages()
        return bool(langs) and str(langs[0]).startswith("zh")
    except Exception:
        return False


def set_language(ui_language: str) -> None:
    """ui_language: auto | en | zh-HK"""
    global _ZH
    if ui_language == "zh-HK":
        _ZH = True
    elif ui_language == "en":
        _ZH = False
    else:
        _ZH = system_prefers_chinese()


def is_chinese() -> bool:
    return _ZH


def tr(key: str, **kwargs) -> str:
    pair = S.get(key)
    if pair is None:
        return key
    text = pair[1] if _ZH else pair[0]
    return text.format(**kwargs) if kwargs else text
