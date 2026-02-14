#!/usr/bin/env bash
set -e

echo "Loading..."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/app"
if [ ! -d "$APP_DIR" ]; then
  echo "Missing app directory: $APP_DIR"
  exit 1
fi
cd "$APP_DIR"

mkdir -p debug temp
export PYTHONPATH="$APP_DIR/src"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Run ./install_dependencies.sh first."
  exit 1
fi

if ! python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
  echo "Python 3.11+ is required. Run ./install_dependencies.sh first."
  exit 1
fi

if ! python3 -c "import importlib.util,sys;mods=['numpy','cv2','PIL','librosa','soundfile','moviepy','imageio','imageio_ffmpeg','PySide6'];missing=[m for m in mods if importlib.util.find_spec(m) is None];raise SystemExit(0 if not missing else 1)" >/dev/null 2>&1; then
  echo "Missing required Python packages. Run ./install_dependencies.sh first."
  exit 1
fi

python3 -m oscimorph > "debug/oscimorph_run.log" 2>&1 &
echo "App launched. Logs: app/debug/oscimorph_run.log"
