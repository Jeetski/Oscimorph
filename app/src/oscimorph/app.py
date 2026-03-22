from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication
from .gui import MainWindow
from .runtime import ensure_user_dirs, package_root


def main() -> int:
    ensure_user_dirs()
    bundled_ffmpeg = package_root() / "vendor" / "ffmpeg" / "ffmpeg.exe"
    if bundled_ffmpeg.exists():
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", str(bundled_ffmpeg))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
