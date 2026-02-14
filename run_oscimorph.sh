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
  echo "Python 3 not found. Please install Python 3.11+ and rerun."
  exit 1
fi

python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
python3 -m pip install -r "$APP_DIR/requirements.txt"

python3 -m oscimorph > "debug/oscimorph_run.log" 2>&1 &
echo "App launched. Logs: app/debug/oscimorph_run.log"
