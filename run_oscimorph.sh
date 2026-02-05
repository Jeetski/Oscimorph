#!/usr/bin/env bash
set -e

echo "Loading..."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p debug temp
export PYTHONPATH="$ROOT_DIR/src"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Please install Python 3.11+ and rerun."
  exit 1
fi

python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
python3 -m pip install -r requirements.txt

python3 -m oscimorph > "debug/oscimorph_run.log" 2>&1 &
echo "App launched. Logs: debug/oscimorph_run.log"
