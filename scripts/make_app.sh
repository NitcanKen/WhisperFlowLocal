#!/bin/bash
# Build a double-clickable WhisperFlow-Local.app into ~/Applications.
# The bundle's main executable is a compiled C shim (proper Mach-O), so
# macOS attributes the Microphone / Accessibility / Input Monitoring
# permissions to "WhisperFlow-Local" — not to python or your terminal.
#
# Optional: pass a source image to (re)generate the app icon:
#   scripts/make_app.sh ~/Downloads/my-logo.png
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "venv missing — run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

APP="$HOME/Applications/WhisperFlow-Local.app"
ICON_SRC="${1:-}"
ICNS="$REPO/scripts/AppIcon.icns"

if [ -n "$ICON_SRC" ]; then
  "$PY" "$REPO/scripts/make_icon.py" "$ICON_SRC" "$ICNS"
fi

# TCC pins ad-hoc grants to the binary's cdhash: if this build produces a
# different hash, existing Accessibility/Input Monitoring approvals keep
# showing ON in System Settings but are silently denied. Capture the old
# hash so we can detect that and reset the stale grants below.
OLD_CDHASH="$(codesign -dvvv "$APP" 2>&1 | awk -F= '/^CDHash=/{print $2}')" || true

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

ICON_KEYS=""
if [ -f "$ICNS" ]; then
  cp "$ICNS" "$APP/Contents/Resources/AppIcon.icns"
  ICON_KEYS="    <key>CFBundleIconFile</key><string>AppIcon</string>"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>WhisperFlow-Local</string>
    <key>CFBundleDisplayName</key><string>WhisperFlow-Local</string>
    <key>CFBundleIdentifier</key><string>com.whisperflow.local</string>
    <key>CFBundleExecutable</key><string>WhisperFlow-Local</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.2.0</string>
    <key>CFBundleVersion</key><string>1.2.0</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>LSUIElement</key><true/>
$ICON_KEYS
    <key>NSMicrophoneUsageDescription</key>
    <string>WhisperFlow-Local records your voice locally to transcribe it. Audio never leaves this Mac.</string>
</dict>
</plist>
PLIST

clang -O2 \
  -DWFL_REPO="\"$REPO\"" \
  -DWFL_PY="\"$PY\"" \
  -o "$APP/Contents/MacOS/WhisperFlow-Local" \
  "$REPO/scripts/launcher.c"

codesign --force -s - "$APP" && echo "signed (ad-hoc)"

NEW_CDHASH="$(codesign -dvvv "$APP" 2>&1 | awk -F= '/^CDHash=/{print $2}')"
if [ -n "$OLD_CDHASH" ] && [ "$OLD_CDHASH" != "$NEW_CDHASH" ]; then
  echo "cdhash changed ($OLD_CDHASH -> $NEW_CDHASH): old TCC grants are now stale."
  tccutil reset Accessibility com.whisperflow.local || true
  tccutil reset ListenEvent com.whisperflow.local || true
  echo ">>> Re-grant Accessibility + Input Monitoring to 'WhisperFlow-Local' in System Settings, then relaunch."
fi

# Refresh LaunchServices so Finder/permission lists pick up the new bundle.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true
touch "$APP"

echo "Built: $APP"
echo "Open it with: open \"$APP\"  (or double-click in Finder)"
echo "First run: grant Microphone / Accessibility / Input Monitoring to 'WhisperFlow-Local'."
