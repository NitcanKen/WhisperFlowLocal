# Overlay Redesign — Book-Unfold Animation + Milky-White Theme + Golden Silk Ribbon

Status: **revision 2 implemented + verified (2026-07-08)**

> **Revision 2 addendum (2026-07-08, user-approved).** After live use the
> thick silk ribbon read as too faint on the cream base and the 30 fps
> rumps-timer drive felt choppy. The ribbon is replaced by a **thread-bundle
> waveform** (reference: user's screenshot — many hair-thin strands that
> converge to a single line in silence and fan out / weave when speaking,
> tapering to points at both ends): 7 deep-sepia strands `#5A4632` @0.30
> (1.0 px) + 3 champagne-gold accents `#C9A96A` @0.55 (1.2 px), each with
> deterministic per-strand phase/frequency/amplitude/speed derived from its
> index (no randomness), envelope = smoothed levels × `sin^0.6` end-taper
> window. Rendering moves to a **self-driven 60 fps `NSTimer` in
> `NSRunLoopCommonModes`** owned by the overlay (animation stepped by real
> dt — unfold 0.28 s / fold 0.20 s regardless of frame rate; timer stops
> when fully closed; frame updates skipped when the width is unchanged).
> `tick(state, level)` keeps its signature but now just feeds state/levels
> at ~30 Hz; `app.py` is unchanged. Book-unfold, milky-white theme, amber
> dot and the shimmer-driven processing state carry over.
Owner file: `whisperflow_local/overlay.py` (+ `tests/test_insert_and_overlay.py`)

## 【核心目標】[Core Goal]

Redesign the waveform HUD pill so that:

1. **Entrance/exit is ceremonial** — the pill unfolds horizontally from a dot
   like a book opening (~280 ms, easeOutBack overshoot), and folds back +
   fades on dismiss (~200 ms, easeInCubic). Replaces the current plain
   ~120 ms alpha fade.
2. **Theme is milky-white (米白)** matching the app logo (`scripts/AppIcon.icns`:
   cream base + champagne-gold "W" ribbon) — frosted translucent cream, not
   the current dark `NSVisualEffectMaterialHUDWindow` look.
3. **The waveform is a golden silk ribbon**, not vertical bars — one
   continuous smoothed band whose undulation and thickness follow live mic
   levels, with gradient + highlight + shadow edges for a 3-D satin feel,
   visually echoing the logo's gold W.

## 【成功標準】[Success Criteria]

1. `whisperflow_local/overlay.py` implements the full design below — unfold
   in/out animation, milky-white frosted theme, ribbon waveform for the
   recording state, travelling-wave ribbon for transcribing/formatting,
   amber breathing recording dot. No bars remain.
2. All animation/geometry logic lives in **module-level pure functions**
   (same pattern as existing `push_history`/`display_heights`), each covered
   by unit tests in `tests/test_insert_and_overlay.py`.
3. `.venv/bin/python -m pytest tests/` exits 0 (whole suite, not just the
   overlay file).
4. `.venv/bin/python -m whisperflow_local --selftest` exits 0.
5. `grep -riE "mock|fake|dummy|stub|placeholder|hardcode|lorem|TODO" whisperflow_local/`
   produces no output (repo hygiene rule from CLAUDE.md).
6. Public interface of `WaveformOverlay` is unchanged: `__init__()`,
   `tick(state: str, level: float)`, `is_visible()` — `app.py` must not need
   edits.

Final visual sign-off is manual: relaunch the .app (it execs the live repo —
**no `make_app.sh` rebuild**) and record once. This human step is outside the
/goal condition.

## 【限制條件】[Constraints]

- Only modify `whisperflow_local/overlay.py` and
  `tests/test_insert_and_overlay.py`. Nothing else.
- No Core Animation (`CASpringAnimation`/`CAShapeLayer` etc.) — everything is
  driven by the existing ~30 fps `tick()` from the rumps timer, drawn in
  `drawRect_` with `NSBezierPath`/`NSGradient`. All AppKit calls stay on the
  main thread (callers already guarantee this).
- No new dependencies.
- Keep `push_history`, `display_heights`, `shimmer_heights` and their
  existing tests working (shimmer_heights is reused as the level source for
  the processing-state ribbon).
- Do not run `scripts/make_app.sh` (needless rebuild breaks TCC grants).

## 【相關內容】[Context — read/run these first]

- Read `whisperflow_local/overlay.py` in full before editing (≈185 lines).
  Current structure: pure helpers → `_WaveView(NSView)` with `drawRect_` →
  `WaveformOverlay` controller with `_build()` + `tick()`.
- Read `tests/test_insert_and_overlay.py` for the existing test style.
- Verified baseline (2026-07-08): pill is 300×44 (`PILL_W`, `PILL_H`),
  36 bars, dark HUD material, alpha-only fade via
  `self._alpha += (target - alpha) * 0.28`; suite is green on main.
- `tick(state, level)` is called ~30×/s; visible states are
  `recording | transcribing | formatting`.

### Design detail (approved)

**Unfold animation.** Replace `_alpha`/`_target` with a progress scalar
`p ∈ [0, 1]` plus direction. Opening: `p += 1/9` per tick (~280 ms); closing:
`p -= 1/6` per tick (~200 ms). Reversals mid-animation continue from the
current `p` (rapid press/release must not jump). Per tick:

- `width = PILL_H + (PILL_W − PILL_H) · ease(p)` where `ease` is
  `ease_out_back` when opening (overshoot tuned to ≈3–6 %, i.e. width may
  briefly exceed `PILL_W`) and `ease_in_cubic` of the reversed progress when
  closing. Panel frame is re-centred every tick
  (`x = screen_cx − width/2`) via `setFrame_display_`; height and corner
  radius stay `PILL_H` so it reads as a capsule/dot throughout.
- Alpha ramps 0→1 over the first 30 % of `p` when opening; fades over the
  last 30 % when closing. `orderFrontRegardless` on open start;
  `orderOut_` once closed (`p ≤ 0`).
- Ribbon amplitude is multiplied by `clamp(p, 0, 1)` so the ribbon rises
  after the book has opened and flattens before it folds shut.
- The content `NSVisualEffectView` resizes with the panel (autoresizing
  width); the wave view likewise.

**Golden silk ribbon (recording state).** Level source: existing
`push_history` → `display_heights`, then `smooth_levels` (moving average,
window 3). Build `N ≈ 24` sample points across the drawable width
(16 px side padding, skip the dot zone on the left):

- Centreline: `y_i = mid + A · lvl_i · sin(0.55·i + 0.18·t)` with
  `A ≈ 0.30 · drawable_height` — undulates with voice and travels with tick
  phase `t` so it flows even at steady volume. A small floor
  (`lvl_i = max(lvl_i, 0.06)`) keeps the ribbon gently alive in silence,
  never dead-flat.
- Thickness: `th_i = TH_MIN + lvl_i · (TH_MAX − TH_MIN)` with
  `TH_MIN ≈ 3 px`, `TH_MAX ≈ 16 px` — louder = fuller band.
- A pure helper returns `[(x, y_top, y_bottom), …]`; the top and bottom
  edges are each smoothed by Catmull-Rom → cubic-Bézier conversion (pure
  helper returning control points), joined into one closed path
  (top edge left→right, bottom edge right→left).
- Fill: vertical `NSGradient` light gold `#E8D5A8` → champagne `#C9A96A` →
  deep gold `#A67C3D`, drawn in the ribbon path. Then stroke the top edge
  white @ 0.60 alpha, 1.5 px (satin highlight) and the bottom edge
  `#7A5A2E` @ 0.35 alpha, 1.5 px (shadow). These three layers are the 3-D
  effect; exact constants may be tuned ±20 % if it looks better, but all
  three layers must exist.

**Processing state (transcribing/formatting).** Same ribbon renderer, but
levels come from `shimmer_heights(N, t)` — a low-amplitude travelling wave.
No separate code path in the view beyond the level source.

**Milky-white theme.** Force light appearance on the effect view
(`NSAppearance.appearanceNamed_(NSAppearanceNameVibrantLight)`), keep the
frosted material, and interpose a cream tint layer (`#F2EBDD` @ ≈0.55 alpha,
same rounded-capsule mask) between the effect view and the wave view so the
pill reads 米白 rather than pure white, while staying translucent.

**Recording dot.** Keep position/size (left, 8 px) and the breathing-pulse
alpha, but recolour from `systemRedColor` to warm amber `#D4A24E`.

## 【執行方式】[Execution Mode]

- 思考深度:think hard before editing — the tricky parts are the mid-animation
  reversal state machine and the closed ribbon path winding.
- Plan mode:不需要 — design 已獲批准,單檔修改,直接執行。
- Subagent / 平行:唔開 subagent — 單一檔案、順序工作、需要共享 context。
- 完整度:完成整個 scope(動畫 + 主題 + 絲帶 + dot + 測試)。不得做一半將剩餘
  列為 gaps/TODO 收工;只有遇到真正 blocker 先可以提早停。

## 【驗證與停止規則】[Verification & Stopping Rules]

- Every completion claim needs fresh verification pasted into the
  conversation: the pytest run, the `--selftest` run, and the hygiene grep.
  Any failure → not done; list the failing command, the error, what was
  fixed, and what remains.
- If the same blocker (missing framework symbol, PyObjC quirk, permission)
  recurs 3 times, stop autonomous changes and output a blocker report:
  steps tried, evidence, remaining options.
- Update the **Status** line at the top of this document to
  `implemented + verified` as the final step.
