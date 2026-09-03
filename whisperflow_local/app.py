"""WhisperFlow-Local menu-bar application (rumps).

All capture/ASR/LLM work runs on worker threads; the AppKit main thread only
renders menu state via a 33 ms timer, so the UI never blocks. Display text is
localized (en / zh-HK) via i18n.tr; internal keys stay English.
"""
import datetime
import fcntl
import json
import os
import queue
import subprocess
import threading
import time

import rumps

from . import APP_NAME, launchagent, paths, permissions
from .applog import log
from .asr import RemoteQwenASRBackend, SenseVoiceEngine
from .asr_router import ASRRouter
from .audio import Recorder, duration_seconds, save_wav
from .clarify import CLARIFY_TIMEOUT, ClarifyPanel, ClarifyRequest
from .config import Config
from .frontmost import frontmost_app_name
from .history import History
from .hotkeys import HotkeyManager, retarget_hold
from .i18n import set_language, tr
from .keycap import KeyCapturePanel, pretty_combo, pretty_key
from .injector import copy_to_clipboard, delete_chars, insert, press_enter
from .llm import LLMUnavailable, PROFILES, VLLMBackend
from .overlay import WaveformOverlay
from .router import LLMRouter
from .sounds import play
from .textproc import (
    apply_dictionary,
    apply_edits,
    apply_phonetic_hotwords,
    guard_verbatim,
    parse_vocab_entry,
    parse_voice_commands,
    quick_clean,
    should_use_llm,
    strip_punctuation,
    to_hk,
    vocab_terms,
)

LANG_CODES = ["auto", "yue", "en", "mixed"]
PROFILE_NAMES = ["Verbatim", "Structured"]
UI_LANGS = [("auto", "Auto (跟隨系統)"), ("en", "English"), ("zh-HK", "廣東話 zh-HK")]
LEVEL_BARS = "▁▂▃▄▅▆▇█"


def _notify(title: str, message: str) -> None:
    try:
        rumps.notification(APP_NAME, title, message)
    except Exception:
        # Notifications need an app bundle; fall back to log output.
        print(f"[{APP_NAME}] {title}: {message}")


class WhisperFlowApp(rumps.App):
    def __init__(self):
        super().__init__("🎤", quit_button=None)
        paths.ensure_dirs()
        if not os.environ.get("WFL_SELFTEST"):
            # --selftest only builds the menu and exits; it must not clash with
            # an already-running app instance over the single-instance lock.
            self._acquire_single_instance_lock()
        self.config = Config()
        set_language(self.config.get("ui_language"))
        self.recorder = Recorder()
        self.asr = self._build_asr_router()
        self.history = History()
        self.llm = self._build_router()

        self.state = "loading"  # loading|idle|recording|transcribing|formatting|error
        self.state_msg = tr("status_loading_model")
        self._last_inserted = ""
        self._last_insert_path = ""
        self._last_formatted = ""
        self._history_dirty = True
        # Set by the LLMRouter / ASRRouter notify callbacks (worker thread); the
        # main-thread UI timer picks them up and shows the fallback/reconnect notice.
        self._llm_switch_dirty = False
        self._llm_switch_note = None
        self._asr_switch_dirty = False
        self._asr_switch_note = None
        self._jobs = queue.Queue()
        self.overlay = WaveformOverlay()
        self.keycap = KeyCapturePanel()
        self.clarify = ClarifyPanel()
        # Worker -> main handoff for the clarify round (same dirty-flag
        # pattern as _llm_switch_note/_llm_switch_dirty above).
        self._clarify_req = None
        self._clarify_dirty = False
        # Set once the card starts closing; the request is resolved only after
        # it has finished, so the worker's Cmd+V can never land in our panel
        # and the closing animation is actually seen.
        self._clarify_pending = None
        self._session_mode = "dictate"
        self._capture_session = None
        self._capture_kind = None
        self._capture_deadline = 0.0
        self._capture_flash_until = 0.0
        self._shown_title = None
        self._shown_status = None

        self._build_menu()

        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

        if os.environ.get("WFL_SELFTEST"):
            return  # construction-only check: skip model load, hotkeys, timers

        if os.environ.get("WFL_NO_PRELOAD"):
            # Test hook: skip eager model load (it loads lazily on first use).
            self.state = "idle"
            self.state_msg = tr("status_ready")
        else:
            self._loader = threading.Thread(target=self._preload, daemon=True)
            self._loader.start()

        self.hotkeys = HotkeyManager(
            self.config.get("ptt_key"),
            self.config.get("toggle_hotkey"),
            self.config.get("generate_hotkey"),
            on_ptt_down=self._begin_recording,
            on_ptt_up=self._end_recording,
            on_toggle=self._toggle_recording,
            on_ptt_mode=self._recording_mode_changed,
        )
        self.hotkeys.start()

        # 30 fps: drives both the menu state and the smooth waveform HUD.
        self._timer = rumps.Timer(self._refresh_ui, 0.033)
        self._timer.start()

        # Hide the Dock icon AFTER the status item exists. Setting the
        # Accessory policy before run() prevents the NSStatusItem from ever
        # being created on this macOS; switching post-launch is safe.
        self._policy_timer = rumps.Timer(self._hide_dock_icon, 1.5)
        self._policy_timer.start()

        if not permissions.accessibility_trusted():
            log("perm", "Accessibility NOT granted — auto-paste disabled, "
                        "falling back to clipboard mode")
            _notify(tr("notify_perm_title"), tr("notify_perm_body"))

        if not self.config.get("onboarded"):
            self._onboard_timer = rumps.Timer(self._run_onboarding_once, 1.0)
            self._onboard_timer.start()

    def _acquire_single_instance_lock(self):
        self._lockfile = open(os.path.join(paths.APP_SUPPORT, "app.lock"), "w")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"{APP_NAME} is already running — exiting this copy.")
            raise SystemExit(0)

    # ------------------------------------------------------------ menu setup
    def _build_menu(self):
        self.item_status = rumps.MenuItem(tr("status_starting"))
        self.item_status.set_callback(None)

        self.item_toggle = rumps.MenuItem(tr("menu_toggle"),
                                          callback=self._menu_toggle)

        self.menu_profile = rumps.MenuItem(tr("menu_profile"))
        self._profile_items = {}
        for name in PROFILE_NAMES:
            item = rumps.MenuItem(tr(f"profile_{name}"),
                                  callback=self._set_profile)
            item._profile_key = name
            self._profile_items[name] = item
            self.menu_profile.add(item)

        self.menu_lang = rumps.MenuItem(tr("menu_language"))
        self._lang_items = {}
        for code in LANG_CODES:
            item = rumps.MenuItem(tr(f"lang_{code}"),
                                  callback=self._set_language)
            item._lang_code = code
            self._lang_items[code] = item
            self.menu_lang.add(item)

        self.menu_engine = rumps.MenuItem(tr("menu_engine"))
        self._engine_items = {}
        for code in ("sensevoice", "qwen3"):
            item = rumps.MenuItem(tr(f"engine_{code}"),
                                  callback=self._set_engine)
            item._engine_code = code
            self._engine_items[code] = item
            self.menu_engine.add(item)

        self.item_llm = rumps.MenuItem(tr("menu_llm"), callback=self._toggle_llm)

        self.menu_model = rumps.MenuItem(tr("menu_model"))
        self._backend_items = {}
        for code in ("remote",):
            item = rumps.MenuItem(tr(f"model_{code}"), callback=self._set_backend)
            item._backend_code = code
            self._backend_items[code] = item
            self.menu_model.add(item)

        self.menu_history = rumps.MenuItem(tr("menu_history"))
        seed = rumps.MenuItem(tr("menu_history_empty"))
        seed.set_callback(None)
        self.menu_history.add(seed)

        self.menu_settings = rumps.MenuItem(tr("menu_settings"))
        self.menu_settings.add(rumps.MenuItem(tr("menu_set_ptt"),
                                              callback=self._set_ptt))
        self.menu_settings.add(rumps.MenuItem(tr("menu_set_toggle"),
                                              callback=self._set_toggle_hotkey))
        self.menu_vocab = rumps.MenuItem(tr("menu_vocab"))
        seed = rumps.MenuItem("…")
        seed.set_callback(None)
        self.menu_vocab.add(seed)
        self._vocab_dirty = True
        self.menu_settings.add(self.menu_vocab)
        self.item_copy_only = rumps.MenuItem(tr("menu_copy_only"),
                                             callback=self._toggle_copy_only)
        self.menu_settings.add(self.item_copy_only)
        self.item_punct = rumps.MenuItem(tr("menu_punct"),
                                         callback=self._toggle_punct)
        self.menu_settings.add(self.item_punct)
        self.item_sounds = rumps.MenuItem(tr("menu_sounds"),
                                          callback=self._toggle_sounds)
        self.menu_settings.add(self.item_sounds)
        self.item_login = rumps.MenuItem(tr("menu_login"),
                                         callback=self._toggle_login)
        self.menu_settings.add(self.item_login)

        self.menu_ui_lang = rumps.MenuItem(tr("menu_ui_lang"))
        self._ui_lang_items = {}
        for code, label in UI_LANGS:
            item = rumps.MenuItem(label, callback=self._set_ui_lang)
            item._ui_lang = code
            self._ui_lang_items[code] = item
            self.menu_ui_lang.add(item)
        self.menu_settings.add(self.menu_ui_lang)

        self.menu_settings.add(rumps.MenuItem(tr("menu_open_config"),
                                              callback=self._open_config))

        self.menu = [
            self.item_status,
            None,
            self.item_toggle,
            self.menu_profile,
            self.menu_lang,
            self.menu_engine,
            self.item_llm,
            self.menu_model,
            self.menu_history,
            None,
            self.menu_settings,
            rumps.MenuItem(tr("menu_permissions"),
                           callback=self._show_onboarding),
            None,
            rumps.MenuItem(tr("menu_quit"), callback=self._quit),
        ]
        self._sync_checkmarks()

    def _sync_checkmarks(self):
        profile = self.config.get("profile")
        for name, item in self._profile_items.items():
            item.state = 1 if name == profile else 0
        lang = self.config.get("language")
        for code, item in self._lang_items.items():
            item.state = 1 if code == lang else 0
        engine = self.config.get("asr_engine")
        for code, item in self._engine_items.items():
            item.state = 1 if code == engine else 0
        ui_lang = self.config.get("ui_language")
        for code, item in self._ui_lang_items.items():
            item.state = 1 if code == ui_lang else 0
        self.item_llm.state = 1 if self.config.get("llm_enabled") else 0
        backend = self.config.get("llm_backend")
        for code, item in self._backend_items.items():
            item.state = 1 if code == backend else 0
        self.item_copy_only.state = 1 if self.config.get("copy_only") else 0
        self.item_punct.state = 1 if self.config.get("punctuation") else 0
        self.item_sounds.state = 1 if self.config.get("sounds") else 0
        self.item_login.state = 1 if launchagent.is_enabled() else 0

    def _hide_dock_icon(self, timer):
        timer.stop()
        try:
            from AppKit import (
                NSApplication,
                NSApplicationActivationPolicyAccessory,
            )
            NSApplication.sharedApplication().setActivationPolicy_(
                NSApplicationActivationPolicyAccessory
            )
            log("ui", "Dock icon hidden (accessory policy applied post-launch)")
        except Exception as exc:
            log("ui", f"policy switch skipped: {exc}")

    # ------------------------------------------------------------ UI refresh
    def _refresh_ui(self, _timer):
        # Smooth HUD first — it wants every tick.
        self.overlay.tick(self.state, self.recorder.level)
        self._poll_capture()
        self._poll_clarify()

        if self.state == "recording":
            bar = LEVEL_BARS[min(len(LEVEL_BARS) - 1,
                                 int(self.recorder.level * len(LEVEL_BARS)))]
            new_title = f"🔴{bar}"
        else:
            new_title = {
                "loading": "⏳", "idle": "🎤", "transcribing": "✍️",
                "formatting": "✨", "generating": "✨", "clarifying": "❓",
                "error": "⚠️",
            }.get(self.state, "🎤")
        # Only touch NSStatusItem/menu when something actually changed.
        if new_title != self._shown_title:
            self._shown_title = new_title
            self.title = new_title
        status = self.state_msg[:70]
        if status != self._shown_status:
            self._shown_status = status
            self.item_status.title = status
        if self._history_dirty:
            self._history_dirty = False
            self._rebuild_history_menu()
        if self._vocab_dirty:
            self._vocab_dirty = False
            self._rebuild_vocab_menu()
        if self._clarify_dirty:
            self._clarify_dirty = False
            self._show_clarify(self._clarify_req)
        if self._llm_switch_dirty:
            self._llm_switch_dirty = False
            self._apply_llm_switch(*self._llm_switch_note)
        if self._asr_switch_dirty:
            self._asr_switch_dirty = False
            self._apply_asr_switch(*self._asr_switch_note)

    def _rebuild_history_menu(self):
        try:
            self.menu_history.clear()
        except AttributeError:
            return  # submenu not materialized yet; retry on next dirty tick
        entries = self.history.recent(10)
        if not entries:
            empty = rumps.MenuItem(tr("menu_history_empty"))
            empty.set_callback(None)
            self.menu_history.add(empty)
        for e in entries:
            ts = datetime.datetime.fromtimestamp(e["ts"]).strftime("%H:%M")
            label = e["formatted"].replace("\n", " ")[:40] or "…"
            item = rumps.MenuItem(f"{ts}  {label}", callback=self._reinsert)
            item._entry_text = e["formatted"]
            self.menu_history.add(item)
        self.menu_history.add(rumps.separator)
        self.menu_history.add(rumps.MenuItem(tr("menu_copy_last"),
                                             callback=self._copy_last))
        self.menu_history.add(rumps.MenuItem(tr("menu_clear_history"),
                                             callback=self._clear_history))

    # ------------------------------------------------------------ recording
    def _begin_recording(self, mode: str = "dictate"):
        if self.state in ("recording", "clarifying"):
            return
        try:
            self.recorder.start()
            self._session_mode = mode
            self.state = "recording"
            self.state_msg = tr("status_recording_gen" if mode == "generate"
                                else "status_recording")
            play("start", self.config.get("sounds"))
        except Exception as exc:
            self.state = "error"
            self.state_msg = f"{tr('notify_mic_title')}: {exc}"
            play("error", self.config.get("sounds"))
            _notify(tr("notify_mic_title"), tr("notify_mic_body"))

    def _end_recording(self, mode: str = "dictate"):
        if self.state != "recording":
            return
        audio = self.recorder.stop()
        play("stop", self.config.get("sounds"))
        if duration_seconds(audio) < 0.3:
            self.state = "idle"
            self.state_msg = tr("status_too_short")
            return
        self.state = "transcribing"
        self.state_msg = tr("status_transcribing")
        # Generation is a different pipeline (ASR -> LLM writes the text ->
        # paste); it must not re-enter the formatting-profile logic.
        self._jobs.put(("generate" if mode == "generate" else "transcribe", audio))

    def _recording_mode_changed(self, mode: str):
        """A modifier arrived after the hold started (people press the
        push-to-talk key first and add Shift a moment later). Reflect the
        upgrade so the user can see generation is armed mid-utterance."""
        if self.state != "recording":
            return
        self._session_mode = mode
        self.state_msg = tr("status_recording_gen" if mode == "generate"
                            else "status_recording")

    def _toggle_recording(self):
        # The hands-free hotkey and the menu item always dictate.
        if self.state == "recording":
            self._end_recording("dictate")
        else:
            self._begin_recording("dictate")

    def _menu_toggle(self, _):
        self._toggle_recording()

    # ------------------------------------------------------------ pipeline
    def _preload(self):
        try:
            self.asr.ensure_loaded(
                progress_cb=lambda m: setattr(self, "state_msg", m))
            self.state = "idle"
            self.state_msg = tr("status_ready")
            log("model", f"{self.asr.engine_name} loaded; app ready")
        except Exception as exc:
            self.state = "error"
            self.state_msg = f"{tr('notify_model_fail')}: {exc}"
            log("model", f"load FAILED: {exc}")
            _notify(tr("notify_model_fail"), str(exc))

    def _work_loop(self):
        while True:
            kind, payload = self._jobs.get()
            try:
                if kind == "transcribe":
                    self._process_audio(payload)
                elif kind == "generate":
                    self._process_generation(payload)
            except Exception as exc:
                self.state = "error"
                self.state_msg = f"{tr('notify_pipeline_err')}: {exc}"
                play("error", self.config.get("sounds"))
                _notify(tr("notify_pipeline_err"), str(exc))

    def _process_audio(self, audio):
        wav = save_wav(audio, paths.AUDIO_TMP)
        log("audio", f"{duration_seconds(audio):.1f}s captured")
        raw = self.asr.transcribe(wav, self.config.get("language"),
                                  context=self._vocab_list())
        # engine_name is read after transcribe so a mid-call fallback shows.
        log("asr", f"({self.asr.engine_name}) "
                   + (raw if raw.strip() else "(empty)"))
        if not raw.strip():
            self.state = "idle"
            self.state_msg = tr("status_heard_nothing")
            return

        text = apply_dictionary(raw, self.config.get("dictionary"))
        parsed = parse_voice_commands(text)

        if parsed.scratch:
            if self._last_inserted and self._last_insert_path in ("paste", "type"):
                delete_chars(len(self._last_inserted))
                self.state_msg = tr("status_scratched")
            else:
                self.state_msg = tr("status_nothing_to_scratch")
            self._last_inserted = ""
            self.state = "idle"
            return

        app_name = frontmost_app_name()  # recorded in history only
        profile = self.config.get("profile")

        formatted = parsed.text
        hk = self.config.get("traditional_hk")
        if should_use_llm(self.config.get("llm_enabled"), parsed.text):
            self.state = "formatting"
            self.state_msg = f"{tr('status_formatting')} ({profile})…"
            try:
                if profile == "Verbatim":
                    vocab = self._vocab_list()
                    base = quick_clean(parsed.text, vocab=vocab, hk=hk)
                    # Deterministic hot-word recovery first (SenseVoice has no
                    # model-level biasing), then the LLM for general homophones.
                    base = apply_phonetic_hotwords(base, vocab)
                    # One call, two channels: a whole-utterance repunctuation
                    # (accepted only if guard_verbatim proves it merely deleted
                    # characters and changed marks) plus a homophone edit list.
                    out = self.llm.propose_cleanup(base, vocab=vocab)
                    guarded = guard_verbatim(base, out["clean"])
                    if out["clean"] and guarded is base:
                        log("route", "verbatim rewrite REJECTED by guard")
                    formatted = apply_edits(guarded, out["edits"], vocab=vocab)
                    log("route", f"llm cleanup ({self.llm.model}): "
                                 f"{out['edits'] if out['edits'] else 'no edits'}")
                else:
                    formatted = self.llm.format_text(parsed.text, profile,
                                                     vocab=self._vocab_list())
                    if hk:
                        formatted = to_hk(formatted)
                    log("route", f"llm rewrite ({profile}, {self.llm.model})")
            except LLMUnavailable as exc:
                formatted = apply_phonetic_hotwords(
                    quick_clean(parsed.text, vocab=self._vocab_list(), hk=hk),
                    self._vocab_list())
                log("route", f"quick_clean + hotwords fallback — {exc}")
                self.state_msg = tr("status_llm_off_raw", err=exc)
                _notify(tr("notify_llm_unavailable_title"), str(exc))
        elif parsed.text:
            # LLM toggled off: deterministic cleanup + hot-word recovery so
            # 繁體/spacing and known-term homophones still get fixed offline.
            formatted = apply_phonetic_hotwords(
                quick_clean(parsed.text, vocab=self._vocab_list(), hk=hk),
                self._vocab_list())
            log("route", "quick_clean + hotwords (LLM disabled)")

        # Structured's punctuation IS its structure — stripping it would
        # destroy the headings and numbered lists, so the toggle is Verbatim-only.
        if (not self.config.get("punctuation") and formatted
                and profile == "Verbatim"):
            formatted = strip_punctuation(formatted)

        if formatted:
            path = insert(formatted, copy_only=self.config.get("copy_only"))
            log("insert", f"path={path} text={formatted[:60]!r}")
            if parsed.press_enter and path in ("paste", "type"):
                press_enter()
            self._last_inserted = formatted
            self._last_insert_path = path
            self._last_formatted = formatted
            self.history.add(raw, formatted, app_name, profile)
            self._history_dirty = True
            shown = formatted.replace("\n", " ")[:48]
            if path == "clipboard-no-perm":
                self.state_msg = tr("status_copied_no_perm")
                _notify(tr("notify_copied_title"), tr("notify_copied_body"))
            else:
                self.state_msg = f"{tr('status_inserted_via')} ({path}): {shown}"
        play("done", self.config.get("sounds"))
        if self.state != "error":
            self.state = "idle"

    def _process_generation(self, audio):
        """Content-generation pipeline: ASR captures the REQUEST, the LLM
        writes the content, and the result is pasted like a dictation.

        Unlike _process_audio there is no deterministic degradation: without
        the LLM there is nothing to paste, so a failure reports and inserts
        NOTHING rather than pasting the user's own request into their app.
        """
        wav = save_wav(audio, paths.AUDIO_TMP)
        log("audio", f"{duration_seconds(audio):.1f}s captured (generate)")
        raw = self.asr.transcribe(wav, self.config.get("language"),
                                  context=self._vocab_list())
        log("asr", f"({self.asr.engine_name}) request: "
                   + (raw if raw.strip() else "(empty)"))
        if not raw.strip():
            self.state = "idle"
            self.state_msg = tr("status_heard_nothing")
            return

        request = parse_voice_commands(
            apply_dictionary(raw, self.config.get("dictionary"))).text
        if not request:
            self.state = "idle"
            self.state_msg = tr("status_heard_nothing")
            return

        app_name = frontmost_app_name()
        vocab = self._vocab_list()
        self.state = "generating"
        self.state_msg = tr("status_generating")
        try:
            questions = self.llm.plan_generation(request, vocab=vocab)
            answers = []
            if questions:
                answers = self._ask_clarify(questions)
                # Only a panel failure returns None. Esc and a timeout both
                # come back with the answers so far and still generate.
                if answers is None:
                    self.state = "idle"
                    self.state_msg = tr("status_gen_cancelled")
                    return
                self.state = "generating"
                self.state_msg = tr("status_generating")
            out = self.llm.generate(request, questions=questions,
                                    answers=answers, vocab=vocab)
        except LLMUnavailable as exc:
            self.state = "error"
            self.state_msg = tr("status_gen_failed", err=exc)
            play("error", self.config.get("sounds"))
            _notify(tr("notify_llm_unavailable_title"), str(exc))
            return

        if self.config.get("traditional_hk"):
            out = to_hk(out)
        if not out.strip():
            self.state = "error"
            self.state_msg = tr("status_gen_empty")
            play("error", self.config.get("sounds"))
            return

        path = insert(out, copy_only=self.config.get("copy_only"))
        log("insert", f"path={path} generated={out[:60]!r}")
        self._last_inserted = out
        self._last_insert_path = path
        self._last_formatted = out
        self.history.add(request, out, app_name, "Generate")
        self._history_dirty = True
        shown = out.replace("\n", " ")[:48]
        if path == "clipboard-no-perm":
            self.state_msg = tr("status_copied_no_perm")
            _notify(tr("notify_copied_title"), tr("notify_copied_body"))
        else:
            self.state_msg = f"{tr('status_inserted_via')} ({path}): {shown}"
        play("done", self.config.get("sounds"))
        if self.state != "error":
            self.state = "idle"

    def _ask_clarify(self, questions):
        """Worker thread: put the questions on screen and wait for answers.

        Returns the answers given so far — Esc ("skipped") and a timeout both
        still generate, because the spoken request is worth writing from
        either way. None means the panel itself failed. Blocking
        here is safe and intended — this is the worker thread, so the AppKit
        main thread keeps rendering the HUD and the menu throughout.
        """
        req = ClarifyRequest(questions)
        self.state = "clarifying"
        self.state_msg = tr("status_clarifying")
        self._clarify_req = req              # publish before flagging, so the
        self._clarify_dirty = True           # reader never sees a stale slot
        # Backstop in case the main-thread timer has died: the worker must
        # never block forever, whatever happens to the UI.
        if not req.done.wait(CLARIFY_TIMEOUT * len(questions) + 10.0):
            req.resolve("timeout")
        log("generate", f"clarify {req.state}: {req.answers}")
        if req.state == "cancelled":
            return None
        return req.answers

    # ------------------------------------------------------------- clarify
    def _show_clarify(self, req):
        """Main thread: display the current question."""
        if req is None or req.done.is_set():
            return
        q = req.questions[req.index]
        req.state = "shown"
        req.extend_deadline()
        # The panel takes the keyboard, so stop holds/toggles firing beneath it.
        self.hotkeys.set_suppressed(True)
        try:
            # Drop the pill instantly: the card is born at its exact rect and
            # grows out of it, so the two must not animate against each other.
            self.overlay.snap_closed()
            self.clarify.show(q["question"], q["options"],
                              tr("clarify_hint"), tr("clarify_other"))
        except Exception as exc:
            log("clarify", f"panel show failed: {exc!r}")
            self._finish_clarify(req, "cancelled")

    def _poll_clarify(self):
        """Main thread, ~30 Hz. Never blocks."""
        pending = self._clarify_pending
        if pending is not None:
            # Wait out the closing animation before letting the worker paste.
            if self.clarify.is_visible():
                return
            self._clarify_pending = None
            req, state, answers = pending
            self.hotkeys.set_suppressed(False)
            req.resolve(state, answers)
            return
        req = self._clarify_req
        if req is None or req.done.is_set():
            return
        try:
            choice = self.clarify.take_choice()
        except Exception as exc:
            log("clarify", f"poll failed: {exc!r}")
            self._finish_clarify(req, "cancelled")
            return
        if choice is None:
            if req.expired():
                # Timing out is not a cancel either: generate with what we
                # have rather than throwing the user's dictation away.
                self._finish_clarify(req, "timeout", req.answers)
            return
        if choice == "skip":
            # Esc means "stop asking and just write it" — the request is still
            # worth generating from, so this is the timeout path, not a cancel.
            self._finish_clarify(req, "skipped", req.answers)
            return
        if choice.startswith("opt:"):
            answer = req.questions[req.index]["options"][int(choice[4:])]
        else:
            answer = choice.split(":", 1)[1]
        req.answers.append(answer)
        req.index += 1
        if req.index >= len(req.questions):
            self._finish_clarify(req, "answered", req.answers)
        else:
            self._show_clarify(req)

    def _finish_clarify(self, req, state, answers=None):
        """The one place a clarify round ends.

        The ordering is load-bearing: the card must be fully off screen before
        the worker resumes, or the synthesized ⌘V would paste into our own
        panel. So this only STARTS the close; _poll_clarify resolves the
        request once the animation has finished.
        """
        self._clarify_req = None
        try:
            self.clarify.begin_hide()
        except Exception as exc:
            log("clarify", f"hide failed: {exc!r}")
            self.hotkeys.set_suppressed(False)
            req.resolve(state, answers)
            return
        self._clarify_pending = (req, state, answers)

    # ------------------------------------------------------------ callbacks
    def _set_profile(self, sender):
        self.config.set("profile", sender._profile_key)
        self._sync_checkmarks()

    def _set_language(self, sender):
        self.config.set("language", sender._lang_code)
        self._sync_checkmarks()

    def _set_engine(self, sender):
        self.config.set("asr_engine", sender._engine_code)
        self.asr.set_engine(sender._engine_code)
        if sender._engine_code == "sensevoice":
            self.menu_engine.title = tr("menu_engine")  # clear any fallback tag
        self._sync_checkmarks()
        # Warm the newly selected engine off the main thread.
        threading.Thread(target=self._preload, daemon=True).start()

    def _build_asr_router(self):
        c = self.config
        local = SenseVoiceEngine()
        remote = RemoteQwenASRBackend(
            c.get("qwen_asr_url"), c.get("qwen_asr_model"),
            connect_timeout=c.get("qwen_asr_connect_timeout"),
            total_timeout=c.get("qwen_asr_total_timeout"),
            api_key=(os.environ.get("WHISPERFLOW_QWEN_ASR_API_KEY")
                     or c.get("qwen_asr_api_key")),
        )
        # asr_engine "qwen3" -> remote-primary auto; "sensevoice" -> local-only.
        backend = "auto" if c.get("asr_engine") == "qwen3" else "local"
        return ASRRouter(
            local=local, remote=remote, backend=backend,
            threshold=c.get("fallback_threshold"),
            cooldown=c.get("fallback_cooldown"),
            notify=self._on_asr_switch,
        )

    def _on_asr_switch(self, event, engine):
        """ASRRouter callback (worker thread) — defer all UI to the timer."""
        self._asr_switch_note = (event, engine)
        self._asr_switch_dirty = True

    def _apply_asr_switch(self, event, engine):
        """Main-thread: reflect an ASR breaker trip / reconnect in menu + notice."""
        if event == "fallback":
            self.menu_engine.title = tr("menu_engine") + tr("engine_fallback_tag")
            self.state_msg = tr("status_asr_fallback")
            _notify(tr("notify_asr_fallback_title"), tr("notify_asr_fallback_body"))
        else:  # reconnected
            self.menu_engine.title = tr("menu_engine")
            self.state_msg = tr("status_asr_reconnected")
            _notify(tr("notify_asr_reconnect_title"), tr("notify_asr_reconnect_body"))

    def _build_router(self):
        c = self.config
        remote = VLLMBackend(
            c.get("vllm_url"), c.get("vllm_model"),
            connect_timeout=c.get("vllm_connect_timeout"),
            ttft_timeout=c.get("vllm_ttft_timeout"),
            total_timeout=c.get("vllm_total_timeout"),
            api_key=(os.environ.get("WHISPERFLOW_VLLM_API_KEY")
                     or c.get("vllm_api_key")),
            reasoning_effort=c.get("vllm_reasoning_effort"),
        )
        return LLMRouter(
            local=None, remote=remote,
            backend="remote",
            threshold=c.get("fallback_threshold"),
            cooldown=c.get("fallback_cooldown"),
            notify=self._on_llm_switch,
        )

    def _set_backend(self, sender):
        self.config.set("llm_backend", sender._backend_code)
        self.llm.set_backend(sender._backend_code)
        if sender._backend_code in ("remote", "local"):
            self.menu_model.title = tr("menu_model")  # clear any fallback tag
        self._sync_checkmarks()

    def _on_llm_switch(self, event, model):
        """LLMRouter callback (worker thread) — defer all UI to the timer."""
        self._llm_switch_note = (event, model)
        self._llm_switch_dirty = True

    def _apply_llm_switch(self, event, model):
        """Main-thread: reflect a breaker trip / reconnect in menu + notice."""
        if event == "fallback":
            self.menu_model.title = tr("menu_model") + tr("model_fallback_tag")
            self.state_msg = tr("status_llm_fallback", model=model)
            _notify(tr("notify_llm_fallback_title"),
                    tr("notify_llm_fallback_body", model=model))
        else:  # reconnected
            self.menu_model.title = tr("menu_model")
            self.state_msg = tr("status_llm_reconnected", model=model)
            _notify(tr("notify_llm_reconnect_title"),
                    tr("notify_llm_reconnect_body", model=model))

    def _set_ui_lang(self, sender):
        self.config.set("ui_language", sender._ui_lang)
        self._sync_checkmarks()
        rumps.alert(APP_NAME, tr("dlg_restart_lang"))

    def _toggle_llm(self, _):
        self.config.set("llm_enabled", not self.config.get("llm_enabled"))
        self._sync_checkmarks()

    def _toggle_copy_only(self, _):
        self.config.set("copy_only", not self.config.get("copy_only"))
        self._sync_checkmarks()

    def _toggle_punct(self, _):
        self.config.set("punctuation", not self.config.get("punctuation"))
        self._sync_checkmarks()

    def _toggle_sounds(self, _):
        self.config.set("sounds", not self.config.get("sounds"))
        self._sync_checkmarks()

    def _toggle_login(self, _):
        launchagent.toggle()
        self._sync_checkmarks()

    def _reinsert(self, sender):
        insert(sender._entry_text, copy_only=self.config.get("copy_only"))
        self.state_msg = tr("status_reinserted")

    def _copy_last(self, _):
        entries = self.history.recent(1)
        if entries:
            copy_to_clipboard(entries[0]["formatted"])
            self.state_msg = tr("status_copied_last")

    def _clear_history(self, _):
        self.history.clear()
        self._history_dirty = True

    # ------------------------------------------------------------ settings
    def _set_ptt(self, _):
        self._start_capture("ptt")

    def _set_toggle_hotkey(self, _):
        self._start_capture("toggle")

    def _start_capture(self, kind: str):
        """Open the capture HUD and arm the one-shot key recorder."""
        self._capture_kind = kind
        self._capture_session = self.hotkeys.begin_capture(kind)
        self._capture_deadline = time.time() + 10.0
        self._capture_flash_until = 0.0
        title = tr("cap_title_ptt" if kind == "ptt" else "cap_title_toggle")
        prompt = tr("cap_prompt" if kind == "ptt" else "cap_prompt_combo")
        log("capture", f"begin kind={kind}")
        try:
            self.keycap.show(title, prompt)
        except Exception as exc:
            log("capture", f"panel show failed: {exc!r}")

    def _poll_capture(self):
        """Advance the capture flow; runs on the main thread (rumps timer)."""
        if self._capture_flash_until:
            if time.time() > self._capture_flash_until:
                self._capture_flash_until = 0.0
                self.keycap.hide()
            return
        session = self._capture_session
        if session is None:
            return
        if session.state == "waiting":
            if time.time() > self._capture_deadline:
                log("capture", "deadline expired — no key captured")
                self.hotkeys.cancel_capture()
                self._capture_session = None
                self.keycap.hide()
            return
        self._capture_session = None
        if session.state == "cancelled":
            self.keycap.hide()
            return
        result = session.result
        # Apply the binding + show the "Saved" flash defensively: if any of this
        # raises, the auto-hide MUST still arm below, or the capture panel stays
        # on screen forever (the "pressed a key, no response" symptom).
        try:
            if self._capture_kind == "ptt":
                # Keep the generation combo pointing at the PTT key when it
                # moves, or it would fire from a key that is no longer PTT.
                gen = retarget_hold(self.config.get("generate_hotkey"),
                                    self.config.get("ptt_key"), result)
                self.hotkeys.update(result, self.config.get("toggle_hotkey"), gen)
                self.config.set("ptt_key", result)
                self.config.set("generate_hotkey", gen)
                shown = pretty_key(result)
            else:
                self.hotkeys.update(self.config.get("ptt_key"), result,
                                    self.config.get("generate_hotkey"))
                self.config.set("toggle_hotkey", result)
                shown = pretty_combo(result)
            self.keycap.show_result(tr("cap_saved", key=shown))
        except Exception as exc:
            log("capture", f"apply/show_result failed: {exc!r}")
        self._capture_flash_until = time.time() + 1.2
        log("capture", f"applied {self._capture_kind}={result!r}")

    # ------------------------------------------------------------ vocabulary
    def _vocab_list(self) -> list:
        return vocab_terms(self.config.get("dictionary"),
                           self.config.get("hotwords"))

    def _rebuild_vocab_menu(self):
        try:
            self.menu_vocab.clear()
        except AttributeError:
            return  # submenu not materialized yet; retry on next dirty tick
        pairs = self.config.get("dictionary") or []
        terms = self.config.get("hotwords") or []
        if not pairs and not terms:
            empty = rumps.MenuItem(tr("vocab_empty"))
            empty.set_callback(None)
            self.menu_vocab.add(empty)
        for i, entry in enumerate(pairs):
            item = rumps.MenuItem(f"{entry.get('from')} → {entry.get('to')}",
                                  callback=self._vocab_delete)
            item._vocab_ref = ("pair", i)
            self.menu_vocab.add(item)
        for i, term in enumerate(terms):
            item = rumps.MenuItem(str(term), callback=self._vocab_delete)
            item._vocab_ref = ("term", i)
            self.menu_vocab.add(item)
        self.menu_vocab.add(rumps.separator)
        self.menu_vocab.add(rumps.MenuItem(tr("vocab_add"),
                                           callback=self._vocab_add))

    def _vocab_add(self, _):
        resp = rumps.Window(
            tr("dlg_vocab_prompt"), APP_NAME, default_text="",
            dimensions=(320, 24),
        ).run()
        if not resp.clicked:
            return
        parsed = parse_vocab_entry(resp.text)
        if parsed is None:
            rumps.alert(APP_NAME, tr("dlg_vocab_invalid"))
            return
        if parsed[0] == "pair":
            entries = list(self.config.get("dictionary") or [])
            entries.append({"from": parsed[1], "to": parsed[2]})
            self.config.set("dictionary", entries)
        else:
            terms = list(self.config.get("hotwords") or [])
            if parsed[1] not in terms:
                terms.append(parsed[1])
            self.config.set("hotwords", terms)
        self._vocab_dirty = True

    def _vocab_delete(self, sender):
        kind, idx = sender._vocab_ref
        if not rumps.alert(APP_NAME, tr("dlg_vocab_delete", item=sender.title),
                           ok=None, cancel=True):
            return
        key = "dictionary" if kind == "pair" else "hotwords"
        entries = list(self.config.get(key) or [])
        if 0 <= idx < len(entries):
            entries.pop(idx)
            self.config.set(key, entries)
        self._vocab_dirty = True

    def _open_config(self, _):
        subprocess.run(["open", paths.APP_SUPPORT], check=False)

    # ------------------------------------------------------------ onboarding
    def _run_onboarding_once(self, timer):
        timer.stop()
        self._show_onboarding(None)
        self.config.set("onboarded", True)

    def _show_onboarding(self, _):
        # Trigger the system prompts so this process appears in the
        # permission lists, then show the live status.
        permissions.prompt_accessibility()
        permissions.request_input_monitoring()
        s = permissions.summary()
        mark = lambda ok: "✅" if ok else "❌"
        body = tr("perm_summary", im=mark(s["input_monitoring"]),
                  ax=mark(s["accessibility"]))
        if not s["accessibility"]:
            body += tr("perm_warning")
        body += tr("onboard_body", ptt=self.config.get("ptt_key"),
                   toggle=self.config.get("toggle_hotkey"))
        rumps.alert(tr("onboard_title", app=APP_NAME), body)
        for pane in ("Privacy_Microphone", "Privacy_Accessibility",
                     "Privacy_ListenEvent"):
            subprocess.run(
                ["open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"],
                check=False,
            )

    def _quit(self, _):
        self.hotkeys.stop()
        rumps.quit_application()


def main():
    # NOTE: do NOT call setActivationPolicy_ before rumps runs — on this
    # macOS it prevents the NSStatusItem from ever being created (verified
    # empirically: 0 windows vs 8). Dock-icon suppression comes from the
    # app bundle's LSUIElement instead, which spawned children inherit.
    WhisperFlowApp().run()
