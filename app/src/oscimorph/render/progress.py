from __future__ import annotations

import os
import tempfile
import time
from typing import Callable

from proglog import ProgressBarLogger


class ProgressTracker:
    def __init__(
        self,
        *,
        render_total: int,
        progress_cb: Callable[[int, int], None] | None,
        progress_log_path: str | None,
    ) -> None:
        self.render_total = max(1, int(render_total))
        self.render_done = 0
        self.audio_total = 0
        self.audio_index = 0
        self.video_total = 0
        self.video_index = 0
        self.progress_cb = progress_cb
        self.progress_log_path = progress_log_path
        self._last_percent = -1
        self._last_write = 0.0

    def update_render(self, current: int) -> None:
        self.render_done = min(self.render_total, int(current))
        self._emit()

    def update_bar(self, bar: str, attr: str, value: int) -> None:
        if bar == "chunk":
            if attr == "total":
                self.audio_total = int(value)
            elif attr == "index":
                self.audio_index = int(value)
        elif bar == "frame_index":
            if attr == "total":
                self.video_total = int(value)
            elif attr == "index":
                self.video_index = int(value)
        self._emit()

    def finish(self) -> None:
        self._write_progress(100)
        if self.progress_cb:
            self.progress_cb(100, 100)

    def _emit(self) -> None:
        total = self.render_total + self.audio_total + self.video_total
        if total <= 0:
            return
        current = self.render_done + self.audio_index + self.video_index
        percent = int((current / total) * 100)
        percent = max(0, min(100, percent))
        if percent < self._last_percent:
            percent = self._last_percent
        if percent != self._last_percent:
            self._last_percent = percent
            if self.progress_cb:
                self.progress_cb(percent, 100)
            self._write_progress(percent)

    def _write_progress(self, percent: int) -> None:
        if not self.progress_log_path:
            return
        now = time.time()
        if percent < 100 and (now - self._last_write) < 0.2:
            return
        self._last_write = now
        with open(self.progress_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"progress: {percent}%\n")


class MoviepyLogger(ProgressBarLogger):
    def __init__(self, tracker: ProgressTracker) -> None:
        super().__init__(min_time_interval=0.2)
        self._tracker = tracker

    def bars_callback(self, bar, attr, value, old_value=None):  # noqa: ANN001
        self._tracker.update_bar(bar, attr, value)


def init_progress_log(output_path: str) -> str | None:
    root_dir = os.getcwd()
    debug_dir = os.path.join(root_dir, "debug")
    try:
        os.makedirs(debug_dir, exist_ok=True)
    except OSError:
        debug_dir = root_dir
    primary = os.path.join(debug_dir, "oscimorph_run.log")
    try:
        with open(primary, "w", encoding="utf-8") as log_file:
            log_file.write("Oscimorph render log\n")
        return primary
    except OSError:
        pass

    fallback = os.path.join(debug_dir, f"oscimorph_run_{int(time.time())}.log")
    try:
        with open(fallback, "w", encoding="utf-8") as log_file:
            log_file.write("Oscimorph render log\n")
        return fallback
    except OSError:
        return None


def ensure_temp_dir(output_path: str) -> None:
    root_dir = os.getcwd()
    temp_dir = os.path.join(root_dir, "temp")
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except OSError:
        return
    os.environ["TMP"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMPDIR"] = temp_dir
    tempfile.tempdir = temp_dir
