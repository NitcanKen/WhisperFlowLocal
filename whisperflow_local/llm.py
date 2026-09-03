"""LLM cleanup layer with two interchangeable backends.

Both backends share all prompt-building (`format_text`, `propose_cleanup`)
in `BaseLLMBackend`; only the wire format differs:

- `OllamaBackend` — the local Ollama `/api/chat` API (`qwen3.5:4b`). Kept as
  `LLMClient` (an alias) for backward compatibility with existing callers/tests.
- `VLLMBackend` — a remote vLLM OpenAI-compatible server (`/v1/chat/completions`,
  e.g. `Qwen3.6-35B`). It **streams** the response and gives up if no first
  token arrives within a short TTFT deadline, so an asleep/overloaded remote
  fails over fast while a legitimately long generation is never cut off.

`router.LLMRouter` supports a GB10-only remote mode used by the app. The
Ollama backend remains only for backward compatibility with older configs and
unit tests; it is not constructed in remote mode.

Qwen3-family models are hybrid reasoning models. Thinking is disabled — on
Ollama via `think: false`, on vLLM via `chat_template_kwargs.enable_thinking`
false — and any `<think>...</think>` block is stripped defensively (measured
60-180 s/utterance with thinking on — unusable for dictation).

Two profiles: Verbatim and Structured.

Verbatim DOES let the model rewrite the whole utterance (an edit list cannot
fix the ASR's misplaced punctuation), but every rewrite must pass
textproc.guard_verbatim — the output's characters, minus punctuation and
case, must be a subsequence of the input's and keep >=70% of them. That
permits deleting fillers and rewriting marks while making translation,
reordering, 書面語 conversion and summarising impossible. Homophone fixes
travel separately as an edit list through textproc.apply_edits' own guards.
Verified against real model output: unguarded rewriting corrupts Cantonese.

Structured is a deliberate transformation and is intentionally unguarded.
"""
import json
import re
import time

import requests

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_COMMON_RULES = (
    "You are a dictation post-processor. The user dictated text that may mix "
    "Cantonese and English in the same sentence (code-switching). "
    "The text comes from automatic speech recognition, so it may contain "
    "recognition errors: a word that sounds like what was said but is wrong "
    "in context — Cantonese homophones or near-homophones, or an English "
    "term transcribed as similar-sounding Chinese characters or as a "
    "different English word. When the sentence context makes the intended "
    "word clear, silently correct it to what the speaker actually said; if "
    "unsure, keep the original word. "
    "NEVER translate between languages; keep every word in the language it was "
    "spoken. Preserve English words embedded in Cantonese sentences exactly. "
    "Return ONLY the processed text with no explanations, no quotes, no labels."
)

# Structured profile: the ONLY prompt that is allowed to rewrite freely.
# Written in Cantonese for the same reason as CLEANUP_SYSTEM below, and
# carrying one full worked example — self-correction, grouping, numbering and
# 口語→書面語 all in one, which is what the model has to get right.
STRUCTURED_SYSTEM = """你係一個廣東話口述整理器。用戶會口頭講一大段嘢（可能中英夾雜、有口水詞、講錯咗會即刻改口）。你要理解成段嘢嘅意思，然後輸出**結構化嘅書面中文**。

規則：
1. 轉做書面語（唔好保留「系、咁、啦、嘅、嗰」等口語字），用香港繁體。
2. 處理講者嘅自我修正：佢改口之後，**只保留最後嗰個版本**，唔好兩個都寫。
3. 有明顯類別就分組（例如「工作」「生活」），每組用數字編號列點。
4. 淨係得一件事就唔好硬砌標題，直接寫一兩句。
5. 英文詞、產品名、地名照原樣保留（Codex、cyberport、thumbnail…），唔好翻譯。
6. 刪走口水詞同重複。
7. **唔准無中生有**：用戶冇講過嘅內容一律唔准加。唔確定就照原意寫。
8. 只輸出整理好嘅內容，唔好加開場白、解釋或者「以下是」之類。

例子：
輸入：今日系五月二十六號，咁我先記錄下，誒，今日有啲乜嘢代辦事項啦。咁個 to do list 系啲乜嘢咧？咁喺工作上啦，咁我想先回復一下琴日嘅一啲合作郵件啦。系啦，咁另外亦都要睇下最近 Codex 有冇啲咩更新嘅新功能啦。咁最後就係去睇一睇 Desktop AI Toolkit 果期既 video 剪完未，順便做埋個 video 嘅 thumbnail。啊，都系唔好啦，今日先唔做封面，留翻俾聽日。另外啦，咁我聽日要去cyberport，系會搭89X ，啊，唔係，268先岩。咁日常生活上咧，咁我想下晝五點鐘去一趟誒 market 系啦。今日我想自己煮飯食。咁第二樣嘢咧，咁就係去水果店度買一啲水果啦。第三樣嘢就係去攞下快遞
輸出：
今天是5月26號，記錄一下今天的代辦事項：

工作：
1. 回覆昨天的合作郵件
2. 看一下最近 Codex 有沒有更新新功能
3. 把 Desktop AI Toolkit 那期影片剪完（影片封面明天再做）
4. 明天坐 268 去 cyberport

生活：
1. 下午五點去一趟超級市場，今天自己煮飯
2. 去水果店買水果
3. 去拿快遞"""


PROFILES = {
    # Verbatim is NOT here: it does not go through format_text at all, it
    # uses the guarded cleanup channel below (CLEANUP_SYSTEM + propose_cleanup
    # + textproc.guard_verbatim), because an unguarded rewrite corrupts
    # Cantonese. Structured is a deliberate transformation, so it has no
    # guard and lives here.
    "Structured": STRUCTURED_SYSTEM,
}


# Verbatim cleanup prompt. Written in Cantonese: the model follows Cantonese
# instructions for Cantonese text far better than English ones (verified
# empirically against the live model).
#
# TWO channels in ONE call, so punctuation repair costs no extra round-trip:
#   "clean" — the whole utterance with ONLY punctuation changed and fillers /
#             stutters deleted. textproc.guard_verbatim proves it did nothing
#             else (skeleton subsequence + length floor) before it is used.
#   "edits" — near-homophone ASR fixes, applied through textproc.apply_edits'
#             existing per-edit guards. Kept separate because a homophone fix
#             CHANGES characters and could never pass the subsequence guard.
CLEANUP_SYSTEM = """你係廣東話聽寫嘅後處理器。輸入係語音識別（ASR）嘅原文，可能廣東話中英夾雜。

你要做兩件事，輸出一個 JSON：

{"clean": "整段修好嘅文字", "edits": [{"from": "錯嘅片段", "to": "正確片段"}]}

【clean】把成段文字重寫一次，但**只准做以下三樣嘢**：
1. 修標點：ASR 亂咁加嘅句號同問號要改返啱。唔係疑問句就唔准用問號；句子未完就唔好落句號。
2. 刪口水詞：呃、嗯、啊、誒、哦、即係、咁呢、你知道啦、um、uh、er…
   **句子開頭、中間、結尾出現嘅都要刪**，唔淨係刪開頭嗰個。
   注意：「即係」有時係真正嘅意思（「即係話…」＝亦即是），咁就要保留；
   純粹用嚟拖時間嗰啲先刪。
3. 刪重複：講者結巴或者重複講咗兩次嘅字。

**嚴禁**（違反嘅話成段會俾程式丟棄）：
- 唔准加任何原文冇嘅字。
- 唔准翻譯、唔准轉書面語。廣東話口語字（嘅、咁、係、俾、唔、而家、依家、乜嘢）要原封不動。
- 唔准改語序、唔准精簡、唔准摘要。
- 除咗上面三樣，一個字都唔准改（英文大小寫可以改，例如 codex → CodeX）。

【edits】另外搵同音字／近音字錯誤：ASR 將講者講嘅字寫成粵語同音或近音嘅另一個字，令上下文明顯唔通順。
- 「from」必須一字不差咁出現喺**原文**入面，而且至少兩個字（要帶埋上下文，例如「中影夾雜」而唔係「影」）。
- 「to」同「from」讀音要相近（粵拼同音或近音），唔准換做讀音完全唔同嘅字。
- 冇明顯錯就 "edits": []。冇把握就唔好改。寧願唔改，都唔好改錯。

例子：
輸入：呃，今日系五月二十六號，咁我，我先記錄下今日有啲乜嘢代辦事項啦？
輸出：{"clean": "今日系五月二十六號，我先記錄下今日有啲乜嘢代辦事項。", "edits": []}
輸入：嗯，我想講下個 sprint 嘅嘢啊？呃，個 bug 係撳完 submit 會 hang 住。即係，我 debug 咗成日啦。
輸出：{"clean": "我想講下個 sprint 嘅嘢。個 bug 係撳完 submit 會 hang 住。我 debug 咗成日啦。", "edits": []}
輸入：我想食意大利份，同埋一杯凍檸茶。
輸出：{"clean": "我想食意大利份，同埋一杯凍檸茶。", "edits": [{"from": "食意大利份", "to": "食意大利粉"}]}
輸入：幫我 book 三點嘅會議室。
輸出：{"clean": "幫我 book 三點嘅會議室。", "edits": []}"""


def parse_edits(raw: str) -> list:
    """Parse the model's edit-list JSON. Malformed output → no edits,
    never an exception: a broken response must degrade to a no-op."""
    try:
        data = json.loads(raw)
        edits = data.get("edits")
        if not isinstance(edits, list):
            return []
        return [
            {"from": str(e["from"]), "to": str(e["to"])}
            for e in edits
            if isinstance(e, dict) and "from" in e and "to" in e
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def parse_cleanup(raw: str) -> dict:
    """Parse the Verbatim cleanup JSON into {"clean": str, "edits": list}.

    Malformed output degrades to a no-op ({"clean": "", "edits": []}) rather
    than raising: an empty `clean` makes guard_verbatim keep the original.
    """
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"clean": "", "edits": []}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"clean": "", "edits": []}
    clean = data.get("clean")
    return {
        "clean": clean if isinstance(clean, str) else "",
        "edits": parse_edits(raw),
    }


# ---------------------------------------------------------------- generation
# Content generation (hold Shift + the push-to-talk key): the user speaks a
# REQUEST and the model writes the text, instead of the utterance itself being
# transcribed. Two steps, because a vague request is worth one round of
# questions rather than a confidently wrong draft:
#   plan_generation -> {"questions": [...]}  (possibly empty -> write it now)
#   generate        -> the finished text
CLARIFY_SYSTEM = """你要判斷用戶嘅內容生成要求夠唔夠清楚。

只輸出 JSON：{"questions": [{"question": "問題", "options": ["選項1", "選項2"]}]}

規則：
- 要求已經夠清楚就輸出 {"questions": []}。呢個係預設答案 —— 唔好為問而問。
- 只有當某個決定會**明顯改變**成篇嘢（例如用咩語言寫、寫俾邊類對象、要正式定隨意）先至問。
- 最多 2 條問題，每條 2 至 3 個選項。
- 問題同選項要簡短（選項唔好超過 4 個字或者一個英文詞）。
- 唔好問內容細節（例如「你想寫幾長」之類嘅小事），淨係問會改變方向嘅嘢。

例子：
要求：幫我 draft 一封 email 俾 Lulu，內容係我哋完成咗 scenario D 嘅開發，可以進行 proof of concept 測試，歡迎俾 feedback
輸出：{"questions": [{"question": "用咩語言寫？", "options": ["English", "繁體中文"]}]}
要求：用英文寫一段 100 字嘅產品介紹，講我哋個 app 支援廣東話聽寫
輸出：{"questions": []}"""

GENERATE_SYSTEM = """你係寫作助手。用戶會口頭講一個要求，你要**直接寫出佢要嘅內容**。

規則：
- 只輸出成品內容本身。唔好加開場白、解釋、「以下是」、markdown code fence 或者標題標籤。
- 寫 email 就只寫**正文**：唔准加 "Subject:" ／「主旨：」嗰行（用戶通常已經喺
  compose window 入面，主旨行貼落正文會錯位）。除非用戶明確叫你寫主旨。
- **唔准喺輸出入面出現方括號 [ ]**。你唔知用戶叫咩名，所以寫完 "Best regards,"
  就即刻收筆，唔好加 [Your Name]／[Date]／XXX 之類嘅嘢等用戶自己填。
- 跟足用戶指定嘅語言。冇指定就跟佢講嘢嘅語言。
- 唔准無中生有：用戶冇提供嘅事實（日期、金額、人名）唔准杜撰。
- 語氣自然、專業、精簡。"""


MAX_CLARIFY_QUESTIONS = 2
MAX_CLARIFY_OPTIONS = 3


def parse_clarify(raw: str) -> list:
    """Parse the planner's questions. Clamps to <=2 questions with 2..3
    options each and drops malformed entries. Never raises: a broken plan
    means 'no questions', i.e. write it directly."""
    try:
        data = json.loads(raw)
        questions = data.get("questions") if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(questions, list):
        return []
    out = []
    for q in questions[:MAX_CLARIFY_QUESTIONS]:
        if not isinstance(q, dict):
            continue
        text = q.get("question")
        opts = q.get("options")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(opts, list):
            continue
        clean = [o.strip() for o in opts
                 if isinstance(o, str) and o.strip()][:MAX_CLARIFY_OPTIONS]
        if len(clean) < 2:
            continue  # a one-option question is not a choice
        out.append({"question": text.strip(), "options": clean})
    return out



class LLMUnavailable(Exception):
    """Raised when a backend cannot be reached, times out, or the model is
    missing. The router treats this as 'this backend failed, try the next'."""


class BaseLLMBackend:
    """Prompt-building shared by every backend. Subclasses implement only the
    wire format via `_chat` (and `ping`)."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # --- wire format (backend-specific) ------------------------------------
    def _chat(self, system: str, user: str, force_json: bool = False) -> str:
        raise NotImplementedError

    def ping(self) -> bool:
        raise NotImplementedError

    # --- prompt building (shared) ------------------------------------------
    def format_text(self, text: str, profile: str, vocab: list = None) -> str:
        """Apply a formatting profile. Raw or unknown profiles pass through.

        vocab: user's preferred terms/spellings — soft bias so the model
        keeps names like 'WhisperFlow' in their canonical form."""
        system = PROFILES.get(profile)
        if not system or not text.strip():
            return text
        if vocab:
            system += (
                " The user's preferred vocabulary (when the dictation refers "
                "to one of these, use this exact spelling): "
                + ", ".join(vocab) + "."
            )
        out = self._chat(system, text)
        return out if out else text

    def propose_cleanup(self, text: str, vocab: list = None) -> dict:
        """Ask the model to repunctuate + de-filler the utterance and to flag
        ASR homophone slips, in ONE call.

        Returns {"clean": str, "edits": [{"from","to"}, ...]}. The caller runs
        `clean` through textproc.guard_verbatim and `edits` through
        textproc.apply_edits, so neither channel can corrupt the output.
        Raises LLMUnavailable when the backend is down (same contract as
        format_text)."""
        if not text.strip():
            return {"clean": "", "edits": []}
        system = CLEANUP_SYSTEM
        if vocab:
            system += (
                "\n\n用戶常用詞（如果原文有片段係呢啲詞嘅近音誤寫，"
                "改返做呢個寫法）：" + "、".join(vocab)
            )
        # Phonetic hint: give the model each character's Cantonese reading so it
        # judges homophones by sound, not by guessing (phonetic-guided
        # correction beats blind rewrite). Defensive import so a missing
        # ToJyutping never breaks the correction path.
        user = text
        try:
            from .textproc import jyutping_hint
            hint = jyutping_hint(text)
            if hint:
                user = f"{text}\n\n（每個字嘅粵拼讀音，供你判斷同音字用）：{hint}"
        except Exception:
            pass
        return parse_cleanup(self._chat(system, user, force_json=True))

    def plan_generation(self, request: str, vocab: list = None) -> list:
        """Ask whether the request needs clarifying. Returns [] to write now."""
        if not request.strip():
            return []
        return parse_clarify(self._chat(CLARIFY_SYSTEM, request, force_json=True))

    def generate(self, request: str, questions: list = None,
                 answers: list = None, vocab: list = None) -> str:
        """Write the content the user asked for. `answers` (parallel to
        `questions`) folds the clarify round into the prompt."""
        system = GENERATE_SYSTEM
        if vocab:
            system += (
                "\n用戶常用詞（用返呢個寫法）：" + "、".join(vocab)
            )
        user = request
        for q, a in zip(questions or [], answers or []):
            if a:
                user += f"\n（{q['question']} → {a}）"
        return self._chat(system, user)


class OllamaBackend(BaseLLMBackend):
    """Local Ollama `/api/chat` (non-streaming). `think: false` disables Qwen3
    reasoning; if the server rejects that flag we retry without it and strip
    any <think> block defensively."""

    def _chat(self, system: str, user: str, force_json: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # disable Qwen3 hybrid reasoning
            "options": {"temperature": 0.2},
        }
        if force_json:
            payload["format"] = "json"
            payload["options"] = {"temperature": 0}
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            if resp.status_code == 400 and "think" in resp.text.lower():
                # Model/server rejects the think flag: retry without it.
                payload.pop("think", None)
                resp = requests.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
            if resp.status_code == 404:
                raise LLMUnavailable(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"Ollama unreachable at {self.base_url}: {exc.__class__.__name__}"
            ) from exc
        content = THINK_RE.sub("", content)
        return content.strip().strip('"').strip()

    def ping(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/version", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False


# Backward-compatible alias: the local Ollama client used to be the only one.
LLMClient = OllamaBackend


class VLLMBackend(BaseLLMBackend):
    """Remote vLLM OpenAI-compatible `/v1/chat/completions`, streamed.

    The request uses `timeout=(connect_timeout, ttft_timeout)`: the connect
    timeout catches an unreachable/asleep host; the read timeout is the
    time-to-first-token deadline — if the server sends no first byte within it
    (down / crashed / queued), requests raises and we surface LLMUnavailable so
    the router falls back. Once tokens flow, `total_timeout` is a wall-clock
    safety cap on a stream that never ends. Thinking is disabled via the vLLM
    chat-template kwarg, reasoning deltas are excluded from the response, and
    any leaked think block is stripped defensively.
    """

    def __init__(self, base_url: str, model: str,
                 connect_timeout: float = 1.0, ttft_timeout: float = 1.0,
                 total_timeout: float = 30.0, clock=time.monotonic,
                 api_key: str = "", reasoning_effort: str = "none"):
        super().__init__(base_url, model, timeout=total_timeout)
        self.connect_timeout = connect_timeout
        self.ttft_timeout = ttft_timeout
        self.total_timeout = total_timeout
        self._clock = clock
        self.api_key = (api_key or "").strip()
        self.reasoning_effort = (reasoning_effort or "").strip()

    def _auth_headers(self) -> dict:
        headers = {
            "Accept": "text/event-stream",
            # Harmless on normal vLLM deployments and required to keep ngrok's
            # free-tier browser interstitial out of API responses.
            "ngrok-skip-browser-warning": "1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _read_stream(self, resp) -> str:
        """Accumulate the answer from an SSE stream until [DONE]. Enforces the
        wall-clock total cap; ignores malformed lines.

        Prefers `delta.content`. Some reasoning-model deployments (observed on
        DeepSeek-V4-Flash) ignore the non-thinking flags and stream the FINAL
        answer through the reasoning channel with `content` left empty; when no
        content arrives at all we fall back to the reasoning text so the backend
        isn't a silent no-op. `THINK_RE` in `_chat` strips any leaked block."""
        start = self._clock()
        content_parts = []
        reasoning_parts = []
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if self._clock() - start > self.total_timeout:
                    raise LLMUnavailable(
                        f"vLLM stream exceeded {self.total_timeout}s cap"
                    )
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                rpiece = delta.get("reasoning") or delta.get("reasoning_content")
                if rpiece:
                    reasoning_parts.append(rpiece)
        finally:
            resp.close()
        content = "".join(content_parts)
        return content if content.strip() else "".join(reasoning_parts)

    def _chat(self, system: str, user: str, force_json: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "temperature": 0 if force_json else 0.2,
            # Disable thinking in chat templates that support this flag (Qwen3).
            "chat_template_kwargs": {"enable_thinking": False},
            # Deployments that ignore the flag above stream the final answer
            # through the reasoning channel; keep it included so _read_stream can
            # salvage it when content is empty (do NOT suppress it).
            "include_reasoning": True,
        }
        # DeepSeek-V4 and other reasoning models ignore enable_thinking; their
        # kill switch is reasoning_effort="none", which both removes the 6-60 s
        # think latency and returns the answer via content. Omitted when blank
        # so plain non-reasoning servers that reject the field are unaffected.
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload, stream=True,
                headers=self._auth_headers(),
                timeout=(self.connect_timeout, self.ttft_timeout),
            )
            if resp.status_code != 200:
                resp.close()
                raise LLMUnavailable(
                    f"vLLM HTTP {resp.status_code} at {self.base_url}"
                )
            content = self._read_stream(resp)
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"vLLM unreachable/slow at {self.base_url}: {exc.__class__.__name__}"
            ) from exc
        content = THINK_RE.sub("", content)
        return content.strip().strip('"').strip()

    def ping(self) -> bool:
        try:
            requests.get(
                f"{self.base_url}/models", headers=self._auth_headers(),
                timeout=self.connect_timeout + 2,
            ).raise_for_status()
            return True
        except requests.RequestException:
            return False
