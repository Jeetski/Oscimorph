from __future__ import annotations

import os
import sys
import time
from pathlib import Path


APP_NAME = "Oscimorph"


def package_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[2]


def _windows_local_appdata() -> Path | None:
    value = os.environ.get("LOCALAPPDATA")
    if value:
        return Path(value)
    return None


def user_data_root() -> Path:
    local = _windows_local_appdata()
    if local is not None:
        return local / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def ensure_user_dirs() -> None:
    for path in (
        user_data_root(),
        debug_dir(),
        output_dir(),
        temp_dir(),
        user_presets_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)


def assets_dir() -> Path:
    return package_root() / "assets"


def bundled_presets_dir() -> Path:
    return package_root() / "presets"


def scripts_dir() -> Path:
    return package_root() / "scripts"


def debug_dir() -> Path:
    return user_data_root() / "debug"


def output_dir() -> Path:
    return user_data_root() / "output"


def temp_dir() -> Path:
    return user_data_root() / "temp"


def user_presets_dir() -> Path:
    return user_data_root() / "presets"


def resource_path(*parts: str) -> str:
    return str(package_root().joinpath(*parts))


def append_debug_log(message: str) -> None:
    try:
        ensure_user_dirs()
        path = debug_dir() / "oscimorph_run.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass
