#!/bin/bash
# Build an unsigned standalone WhisperFlow-Local.app with pyinstaller.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/pip install pyinstaller >/dev/null

.venv/bin/pyinstaller \
  --windowed \
  --name "WhisperFlow-Local" \
  --osx-bundle-identifier com.whisperflow.local \
  --collect-all funasr \
  --collect-all modelscope \
  --hidden-import whisperflow_local \
  --paths . \
  launcher.py 2>/dev/null || {
    # Generate the tiny launcher on first use, then retry once.
    printf 'from whisperflow_local.app import main\nmain()\n' > launcher.py
    .venv/bin/pyinstaller \
      --windowed \
      --name "WhisperFlow-Local" \
      --osx-bundle-identifier com.whisperflow.local \
      --collect-all funasr \
      --collect-all modelscope \
      --hidden-import whisperflow_local \
      --paths . \
      launcher.py
  }

echo "Built: dist/WhisperFlow-Local.app (unsigned local build)"
