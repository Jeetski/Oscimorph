#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/app"
REQ_FILE="$APP_DIR/requirements.txt"

if [ ! -d "$APP_DIR" ]; then
  echo "Missing app directory: $APP_DIR"
  exit 1
fi
if [ ! -f "$REQ_FILE" ]; then
  echo "Missing requirements file: $REQ_FILE"
  exit 1
fi

need_python_install=0
need_python_upgrade=0
can_auto_python=0
need_ffmpeg=0
can_auto_ffmpeg=0
pip_checks_failed=0

missing_pkgs=()
outdated_pkgs=()
requirements=()

while IFS= read -r line; do
  pkg="$(printf "%s" "$line" | sed -E 's/[<>=!~].*$//' | xargs)"
  if [ -n "$pkg" ]; then
    requirements+=("$pkg")
  fi
done < <(grep -Ev '^[[:space:]]*($|#)' "$REQ_FILE")

contains_pkg() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

python_status="satisfied"
pip_status="satisfied"
ffmpeg_status="satisfied"

if ! command -v python3 >/dev/null 2>&1; then
  need_python_install=1
  python_status="missing"
  pip_status="unknown (python missing)"
else
  if ! python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
    need_python_upgrade=1
    python_status="outdated"
    pip_status="unknown (python outdated)"
  fi
fi

if command -v brew >/dev/null 2>&1 || command -v apt-get >/dev/null 2>&1 || command -v dnf >/dev/null 2>&1 || command -v pacman >/dev/null 2>&1 || command -v zypper >/dev/null 2>&1; then
  can_auto_python=1
  can_auto_ffmpeg=1
fi

if [ "$need_python_install" -eq 0 ] && [ "$need_python_upgrade" -eq 0 ]; then
  if ! python3 -m pip --version >/dev/null 2>&1; then
    pip_checks_failed=1
    pip_status="missing"
  else
    for pkg in "${requirements[@]}"; do
      if ! python3 -m pip show "$pkg" >/dev/null 2>&1; then
        missing_pkgs+=("$pkg")
      fi
    done

    tmp_outdated="$(mktemp)"
    if python3 -m pip list --outdated --format=json >"$tmp_outdated" 2>/dev/null; then
      mapfile -t outdated_pkgs < <(
        python3 - "$tmp_outdated" "${requirements[@]}" <<'PY'
import json
import sys

path = sys.argv[1]
required = {x.lower() for x in sys.argv[2:]}
with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)
for row in rows:
    name = row.get("name", "")
    if name.lower() in required:
        print(name)
PY
      )
    fi
    rm -f "$tmp_outdated"
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  need_ffmpeg=1
  ffmpeg_status="missing"
fi

echo
echo "Dependency check summary:"
echo "-------------------------"
echo "- Python 3.11+: $python_status"
if [ "$need_python_install" -eq 1 ] && [ "$can_auto_python" -eq 1 ]; then
  echo "  Action available: install via package manager"
fi
if [ "$need_python_upgrade" -eq 1 ] && [ "$can_auto_python" -eq 1 ]; then
  echo "  Action available: upgrade via package manager"
fi
if [ "$need_python_install" -eq 1 ] && [ "$can_auto_python" -eq 0 ]; then
  echo "  Action required: manual install"
fi
if [ "$need_python_upgrade" -eq 1 ] && [ "$can_auto_python" -eq 0 ]; then
  echo "  Action required: manual upgrade"
fi

echo "- pip: $pip_status"
if [ "$pip_checks_failed" -eq 1 ]; then
  echo "  Action required: repair pip for current Python install"
fi

echo "- ffmpeg: $ffmpeg_status"
if [ "$need_ffmpeg" -eq 1 ] && [ "$can_auto_ffmpeg" -eq 1 ]; then
  echo "  Action available: install via package manager"
fi
if [ "$need_ffmpeg" -eq 1 ] && [ "$can_auto_ffmpeg" -eq 0 ]; then
  echo "  Action required: manual install"
fi

echo "- Python packages:"
for pkg in "${requirements[@]}"; do
  state="unknown (python/pip unavailable)"
  if [ "$need_python_install" -eq 0 ] && [ "$need_python_upgrade" -eq 0 ] && [ "$pip_checks_failed" -eq 0 ]; then
    if contains_pkg "$pkg" "${missing_pkgs[@]}"; then
      state="missing"
    elif contains_pkg "$pkg" "${outdated_pkgs[@]}"; then
      state="outdated"
    else
      state="satisfied"
    fi
  fi
  echo "  - $pkg: $state"
done

if [ "$need_python_install" -eq 0 ] && [ "$need_python_upgrade" -eq 0 ] && [ "$pip_checks_failed" -eq 0 ] && [ "${#missing_pkgs[@]}" -eq 0 ] && [ "${#outdated_pkgs[@]}" -eq 0 ] && [ "$need_ffmpeg" -eq 0 ]; then
  echo "- Everything required is already installed and up to date."
  exit 0
fi

echo
read -r -p "Proceed with install/update tasks? [y/N] " answer
case "$answer" in
  y|Y|yes|YES) ;;
  *)
    echo "Cancelled by user."
    exit 1
    ;;
esac

install_python() {
  if command -v brew >/dev/null 2>&1; then
    brew install python@3.11
    return $?
  fi
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y python3 python3-pip
    return $?
  fi
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip
    return $?
  fi
  if command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm python python-pip
    return $?
  fi
  if command -v zypper >/dev/null 2>&1; then
    sudo zypper install -y python311 python311-pip
    return $?
  fi
  return 1
}

install_ffmpeg() {
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
    return $?
  fi
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
    return $?
  fi
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y ffmpeg
    return $?
  fi
  if command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm ffmpeg
    return $?
  fi
  if command -v zypper >/dev/null 2>&1; then
    sudo zypper install -y ffmpeg
    return $?
  fi
  return 1
}

if [ "$need_python_install" -eq 1 ] || [ "$need_python_upgrade" -eq 1 ]; then
  if ! install_python; then
    echo "Unable to install/upgrade Python automatically. Please install Python 3.11+ manually."
    exit 1
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is still not available on PATH."
  exit 1
fi

if ! python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
  echo "Python version is still below 3.11."
  exit 1
fi

if ! python3 -m pip install --upgrade pip; then
  echo "Failed to upgrade pip."
  exit 1
fi

to_install=()
for pkg in "${missing_pkgs[@]}" "${outdated_pkgs[@]}"; do
  [ -n "$pkg" ] || continue
  found=0
  for existing in "${to_install[@]}"; do
    if [ "$existing" = "$pkg" ]; then
      found=1
      break
    fi
  done
  if [ "$found" -eq 0 ]; then
    to_install+=("$pkg")
  fi
done

if [ "${#to_install[@]}" -eq 0 ]; then
  to_install=("${requirements[@]}")
fi

if [ "${#to_install[@]}" -gt 0 ]; then
  if ! python3 -m pip install --upgrade "${to_install[@]}"; then
    echo "Failed to install/update required Python packages."
    exit 1
  fi
fi

if [ "$need_ffmpeg" -eq 1 ]; then
  if ! install_ffmpeg; then
    echo "Unable to install ffmpeg automatically. Please install ffmpeg manually and add it to PATH."
    exit 1
  fi
fi

echo
echo "Dependency install/update completed."
exit 0