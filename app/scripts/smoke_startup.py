from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from oscimorph.gui import MainWindow


def main() -> int:
    os.environ["OSCIMORPH_SKIP_STARTUP"] = "1"

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.processEvents()

    QTimer.singleShot(250, app.quit)
    exit_code = app.exec()
    window.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
