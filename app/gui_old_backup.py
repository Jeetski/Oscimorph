from __future__ import annotations

import os
from dataclasses import asdict
import math

from PySide6.QtCore import QThread, Signal, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSplitter,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QGroupBox,
    QStyle,
    QToolButton,
)
from PySide6.QtWidgets import QStyle, QStyleOptionSlider
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

import numpy as np

from .audio import AudioAnalysis, band_at_frame, load_and_analyze
from .render import RenderCancelled, RenderSettings, render_video


class RenderWorker(QThread):
    progress = Signal(int, int)
    finished = Signal()
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, settings: RenderSettings) -> None:
        super().__init__()
        self.settings = settings
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            render_video(
                self.settings,
                progress_cb=self.progress.emit,
                cancel_cb=lambda: self._cancel_requested,
            )
        except RenderCancelled:
            self.cancelled.emit()
            return
        except Exception:  # noqa: BLE001 - pass error to UI
            import traceback

            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit()


class AudioAnalysisWorker(QThread):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, audio_path: str, fps: int, bands: int) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.fps = fps
        self.bands = bands

    def run(self) -> None:
        try:
            analysis = load_and_analyze(self.audio_path, fps=self.fps, bands=self.bands)
        except Exception:  # noqa: BLE001 - pass error to UI
            import traceback

            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(analysis)


class RingPreview(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._color = QColor(0, 255, 204)
        self._band_values: np.ndarray | None = None
        self._energy = 0.0
        self._displace = (0.0, 0.0)
        self._thickness = 2.0
        self._warp_amount = 0.0
        self._warp_phase = 0.0
        self._shape_type = "ring"
        self._polygon_sides = 5
        self._shape_rotation = 0.0
        self.setMinimumSize(480, 320)

    def update_state(
        self,
        *,
        color: QColor,
        band_values: np.ndarray | None,
        energy: float,
        displace_x: float,
        displace_y: float,
        thickness: float,
        warp_amount: float,
        warp_phase: float,
        shape_type: str,
        polygon_sides: int,
        shape_rotation: float,
    ) -> None:
        self._color = color
        self._band_values = band_values
        self._energy = energy
        self._displace = (displace_x, displace_y)
        self._thickness = thickness
        self._warp_amount = warp_amount
        self._warp_phase = warp_phase
        self._shape_type = shape_type
        self._polygon_sides = polygon_sides
        self._shape_rotation = shape_rotation
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.black)

        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        self._draw_grid(painter, w, h)

        cx = w * 0.5 + self._displace[0]
        cy = h * 0.5 + self._displace[1]
        base_radius = min(w, h) * 0.25

        color = QColor(self._color)
        boost = 0.4 + 0.6 * max(0.0, min(1.0, self._energy))
        color.setRed(min(255, int(color.red() * boost)))
        color.setGreen(min(255, int(color.green() * boost)))
        color.setBlue(min(255, int(color.blue() * boost)))

        pen = QPen(color)
        pen.setWidthF(max(1.0, self._thickness))
        painter.setPen(pen)

        rotation = float(self._shape_rotation) * math.pi / 180.0
        if self._shape_type == "polygon":
            sides = max(3, int(self._polygon_sides))
            points = []
            for i in range(sides + 1):
                t = (i / sides) * 2 * math.pi + rotation
                wobble = math.sin(t * 2.0 + self._warp_phase) * self._warp_amount
                radius = base_radius + wobble
                x = cx + math.cos(t) * radius
                y = cy + math.sin(t) * radius
                points.append((x, y))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        else:
            points = []
            steps = 180
            for i in range(steps + 1):
                t = (i / steps) * 2 * math.pi + rotation
                wobble = math.sin(t * 3.0 + self._warp_phase) * self._warp_amount
                radius = base_radius + wobble
                x = cx + math.cos(t) * radius
                y = cy + math.sin(t) * radius
                points.append((x, y))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

    def _draw_grid(self, painter: QPainter, w: int, h: int) -> None:
        painter.save()
        pen = QPen(QColor(255, 255, 255, 22))
        pen.setWidth(1)
        painter.setPen(pen)
        cols = 12
        rows = 8
        for i in range(1, cols):
            x = int((w / cols) * i)
            painter.drawLine(x, 0, x, h)
        for i in range(1, rows):
            y = int((h / rows) * i)
            painter.drawLine(0, y, w, y)
        painter.restore()


class LoopSlider(QSlider):
    loopChanged = Signal(int, int)

    def __init__(self, orientation: Qt.Orientation) -> None:
        super().__init__(orientation)
        self._loop_in = 0
        self._loop_out = 0
        self._drag_mode: str | None = None

    def set_loop_region(self, loop_in: int, loop_out: int) -> None:
        self._loop_in = max(0, int(loop_in))
        self._loop_out = max(self._loop_in, int(loop_out))
        self.update()
        self.loopChanged.emit(self._loop_in, self._loop_out)

    def paintEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        super().paintEvent(event)
        if self.maximum() <= 0:
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        if self._loop_out > self._loop_in:
            span = max(1, self.maximum() - self.minimum())
            start_ratio = (self._loop_in - self.minimum()) / span
            end_ratio = (self._loop_out - self.minimum()) / span
            left = groove.left() + int(groove.width() * start_ratio)
            right = groove.left() + int(groove.width() * end_ratio)

            region_color = QColor(0, 255, 204, 90)
            painter.fillRect(left, groove.top(), max(2, right - left), groove.height(), region_color)

            handle_color = QColor(0, 255, 204, 200)
            handle_width = 6
            painter.fillRect(left - handle_width // 2, groove.top(), handle_width, groove.height(), handle_color)
            painter.fillRect(right - handle_width // 2, groove.top(), handle_width, groove.height(), handle_color)

        playhead_x = self._value_to_pos(self.value(), groove)
        play_color = QColor(255, 160, 0, 220)
        painter.fillRect(playhead_x - 1, groove.top() - 4, 2, groove.height() + 8, play_color)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self
        )
        x = event.position().x()
        left = self._value_to_pos(self._loop_in, groove)
        right = self._value_to_pos(self._loop_out, groove)
        if abs(x - left) <= 6:
            self._drag_mode = "in"
            return
        if abs(x - right) <= 6:
            self._drag_mode = "out"
            return
        self._drag_mode = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        if self._drag_mode:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self
            )
            value = self._pos_to_value(event.position().x(), groove)
            if self._drag_mode == "in":
                self.set_loop_region(min(value, self._loop_out), self._loop_out)
            elif self._drag_mode == "out":
                self.set_loop_region(self._loop_in, max(value, self._loop_in))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        self._drag_mode = None
        super().mouseReleaseEvent(event)

    def _value_to_pos(self, value: int, groove) -> int:  # noqa: ANN001 - Qt signature
        span = max(1, self.maximum() - self.minimum())
        ratio = (value - self.minimum()) / span
        return groove.left() + int(groove.width() * ratio)

    def _pos_to_value(self, x: float, groove) -> int:  # noqa: ANN001 - Qt signature
        span = max(1, groove.width())
        ratio = (x - groove.left()) / span
        value = self.minimum() + int((self.maximum() - self.minimum()) * ratio)
        return max(self.minimum(), min(self.maximum(), value))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Oscimorph")
        self.setMinimumWidth(720)
        self._color_updating = False

        self.media_path = QLineEdit()
        self.audio_path = QLineEdit()
        self.audio_path.editingFinished.connect(self._on_audio_path_committed)
        self.output_path = QLineEdit(os.path.join(os.getcwd(), "output.mp4"))

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(1280)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(720)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.valueChanged.connect(self._schedule_audio_reload)

        self.line_check = QCheckBox("Waveform")
        self.line_check.setChecked(True)

        self.lissajous_check = QCheckBox("Lissajous")
        self.lissajous_check.setChecked(True)

        self.glow_check = QCheckBox("Glow")
        self.glow_check.setChecked(True)

        self.band_count = 5

        self.preserve_check = QCheckBox("Preserve aspect ratio (black bars)")
        self.preserve_check.setChecked(True)

        self.render_mode = QComboBox()
        self.render_mode.addItem("Edge + Overlay", "edge_overlay")
        self.render_mode.addItem("Edge Only", "edge_only")
        self.render_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.edge_method = QComboBox()
        self.edge_method.addItem("Sobel (fast)", "sobel")
        self.edge_method.addItem("Canny (crisp)", "canny")

        self.edge_threshold = QDoubleSpinBox()
        self.edge_threshold.setRange(0.0, 0.5)
        self.edge_threshold.setSingleStep(0.01)
        self.edge_threshold.setValue(0.08)

        self.glow_strength = QDoubleSpinBox()
        self.glow_strength.setRange(0.0, 2.5)
        self.glow_strength.setSingleStep(0.05)
        self.glow_strength.setValue(0.85)

        self.smoothing_check = QCheckBox("Smoothing")
        self.smoothing_check.setChecked(False)
        self.smoothing_amount = QDoubleSpinBox()
        self.smoothing_amount.setRange(0.01, 1.0)
        self.smoothing_amount.setSingleStep(0.05)
        self.smoothing_amount.setValue(0.2)
        self.smoothing_amount.setEnabled(False)
        self.smoothing_check.toggled.connect(self.smoothing_amount.setEnabled)

        self.media_mode = QComboBox()
        self.media_mode.addItem("Media", "media")
        self.media_mode.addItem("Shapes", "shapes")
        self.media_mode.currentIndexChanged.connect(self._on_media_mode_changed)

        self.shape_type = QComboBox()
        self.shape_type.addItem("Ring", "ring")
        self.shape_type.addItem("Polygon", "polygon")
        self.shape_type.currentIndexChanged.connect(self._on_shape_type_changed)

        self.polygon_sides = QSpinBox()
        self.polygon_sides.setRange(3, 12)
        self.polygon_sides.setValue(5)

        self.shape_rotation = QDoubleSpinBox()
        self.shape_rotation.setRange(0.0, 360.0)
        self.shape_rotation.setSingleStep(5.0)
        self.shape_rotation.setValue(0.0)

        self.effect_controls = []
        self.displace_x_amount, self.displace_x_band = self._make_effect_row(0.0, 40.0, 6.0)
        self.displace_y_amount, self.displace_y_band = self._make_effect_row(0.0, 40.0, 6.0)
        self.thickness_amount, self.thickness_band = self._make_effect_row(0.0, 8.0, 3.0)
        self.glow_amount, self.glow_band = self._make_effect_row(0.0, 3.0, 1.0)
        self.threshold_amount, self.threshold_band = self._make_effect_row(0.0, 0.2, 0.05)
        self.warp_amount, self.warp_band = self._make_effect_row(0.0, 40.0, 8.0)
        self.warp_speed_amount, self.warp_speed_band = self._make_effect_row(0.0, 6.0, 2.0)
        self.rotation_mod_amount, self.rotation_mod_band = self._make_effect_row(0.0, 360.0, 0.0)

        self.color_button = QPushButton("Pick Color")
        self.color_button.clicked.connect(self._open_color_dialog)
        self.hex_input = QLineEdit("#00FFCC")
        self.hex_input.editingFinished.connect(self._sync_from_hex)

        self.r_spin = QSpinBox()
        self.g_spin = QSpinBox()
        self.b_spin = QSpinBox()
        for spin in (self.r_spin, self.g_spin, self.b_spin):
            spin.setRange(0, 255)
            spin.valueChanged.connect(self._sync_from_rgb)

        self._set_color(QColor(0, 255, 204))
        self._update_band_selectors()
        self._set_band_defaults()
        self._wire_preview_controls()

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self._on_render)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        self.render_stack = QStackedLayout()
        render_row = QWidget()
        render_layout = QHBoxLayout(render_row)
        render_layout.setContentsMargins(0, 0, 0, 0)
        render_layout.addWidget(self.render_button)
        render_layout.addWidget(self.cancel_button)
        self.render_stack.addWidget(render_row)
        self.render_stack.addWidget(self.progress)
        self.render_stack.setCurrentIndex(0)

        self.preview = RingPreview()
        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.positionChanged.connect(self._on_preview_position)
        self.preview_player.durationChanged.connect(self._on_preview_duration)

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        self.preview_timer.timeout.connect(self._tick_preview)

        self.play_button = QToolButton()
        self.play_button.setToolTip("Play/Pause")
        self.play_button.setAutoRaise(True)
        self.play_button.clicked.connect(self._toggle_preview_play)
        self.stop_button = QToolButton()
        self.stop_button.setToolTip("Stop")
        self.stop_button.setAutoRaise(True)
        self.stop_button.clicked.connect(self._stop_preview)
        self.loop_check = QToolButton()
        self.loop_check.setCheckable(True)
        self.loop_check.setToolTip("Loop")
        self.loop_check.setAutoRaise(True)
        self.mute_button = QToolButton()
        self.mute_button.setCheckable(True)
        self.mute_button.setToolTip("Mute")
        self.mute_button.setAutoRaise(True)
        self.mute_button.toggled.connect(self._toggle_mute)

        self.preview_slider = LoopSlider(Qt.Horizontal)
        self.preview_slider.setRange(0, 0)
        self.preview_slider.sliderPressed.connect(self._on_preview_seek_start)
        self.preview_slider.sliderReleased.connect(self._on_preview_seek_end)
        self.preview_slider.valueChanged.connect(self._on_preview_seek)
        self.preview_slider.loopChanged.connect(self._on_loop_changed)

        self.preview_time = QLabel("0:00 / 0:00")
        self.preview_in_label = QLabel("In: 0:00")
        self.preview_out_label = QLabel("Out: 0:00")
        self.set_in_button = QToolButton()
        self.set_in_button.setToolTip("Set In")
        self.set_in_button.setAutoRaise(True)
        self.set_in_button.clicked.connect(self._set_preview_in)
        self.set_out_button = QToolButton()
        self.set_out_button.setToolTip("Set Out")
        self.set_out_button.setAutoRaise(True)
        self.set_out_button.clicked.connect(self._set_preview_out)
        self._set_media_icons(is_playing=False)

        root = QWidget()
        layout = QHBoxLayout(root)

        preview_panel_widget = QWidget()
        preview_panel = QVBoxLayout(preview_panel_widget)
        preview_header = QHBoxLayout()
        preview_header.addStretch(1)
        preview_label = QLabel("Preview (proxy)\nFinal render uses media or shapes")
        preview_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        preview_header.addWidget(preview_label)
        preview_panel.addLayout(preview_header)
        preview_panel.addWidget(self.preview)
        preview_panel.addLayout(self._build_transport_row())
        preview_panel.addWidget(self.preview_slider)
        preview_panel.addWidget(self.preview_time)
        preview_panel.addLayout(self._build_in_out_row())
        layout.addWidget(preview_panel_widget)

        side_panel_widget = QWidget()
        side_panel = QVBoxLayout(side_panel_widget)
        side_panel.setContentsMargins(0, 0, 0, 0)

        side_panel.addWidget(self._build_framed_section("Inputs / Output", self._build_io_layout()))
        side_panel.addWidget(self._build_framed_section("Render", self._build_render_layout()))
        side_panel.addWidget(self._build_framed_section("Input / Shapes", self._build_shape_layout()))
        side_panel.addWidget(self._build_framed_section("Effects", self._build_effects_layout()))
        side_panel.addWidget(self._build_framed_section("Color", self._build_color_layout()))
        side_panel.addLayout(self.render_stack)
        side_panel.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(side_panel_widget)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(preview_panel_widget)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self.width() // 2, self.width() // 2])
        layout.addWidget(splitter)

        self.setCentralWidget(root)
        self.worker: RenderWorker | None = None
        self.audio_worker: AudioAnalysisWorker | None = None
        self.audio_analysis: AudioAnalysis | None = None
        self._audio_load_token = 0
        self._preview_dragging = False
        self._preview_loop_in = 0
        self._preview_loop_out = 0
        self._preview_smoothed: np.ndarray | None = None
        self._preview_phase = 0.0
        self._set_preview_enabled(False)
        self._on_media_mode_changed()
        self._apply_theme()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        if event.key() == Qt.Key_Escape:
            choice = QMessageBox.question(
                self,
                "Exit Oscimorph",
                "Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice == QMessageBox.Yes:
                self.close()
            else:
                event.ignore()
            return
        super().keyPressEvent(event)

    def _build_io_layout(self) -> QFormLayout:
        form = QFormLayout()

        form.addRow("Media", self._build_media_row())
        form.addRow("MP3", self._with_browse(self.audio_path, self._browse_audio))
        form.addRow("Output", self._with_browse(self.output_path, self._browse_output))

        return form

    def _build_render_layout(self) -> QFormLayout:
        form = QFormLayout()

        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(QLabel("W"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("H"))
        size_layout.addWidget(self.height_spin)
        size_layout.addStretch(1)

        toggles = QWidget()
        toggles_layout = QHBoxLayout(toggles)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.line_check)
        toggles_layout.addWidget(self.lissajous_check)
        toggles_layout.addWidget(self.glow_check)
        toggles_layout.addStretch(1)

        form.addRow("Size", size_row)
        form.addRow("FPS", self.fps_spin)
        form.addRow("Render", toggles)
        form.addRow("Mode", self.render_mode)
        form.addRow("Edge Method", self.edge_method)
        form.addRow("Edge Threshold", self.edge_threshold)
        form.addRow("Glow Strength", self.glow_strength)
        form.addRow("Aspect", self.preserve_check)

        return form

    def _build_shape_layout(self) -> QFormLayout:
        form = QFormLayout()
        form.addRow("Input", self.media_mode)
        form.addRow("Shape", self.shape_type)
        form.addRow("Polygon Sides", self.polygon_sides)
        form.addRow("Orientation", self.shape_rotation)
        return form

    def _build_effects_layout(self) -> QFormLayout:
        form = QFormLayout()
        form.addRow("Smoothing", self._build_smoothing_row())
        form.addRow("Displace X", self._build_effect_row(self.displace_x_amount, self.displace_x_band))
        form.addRow("Displace Y", self._build_effect_row(self.displace_y_amount, self.displace_y_band))
        form.addRow("Thickness Mod", self._build_effect_row(self.thickness_amount, self.thickness_band))
        form.addRow("Glow Mod", self._build_effect_row(self.glow_amount, self.glow_band))
        form.addRow("Threshold Mod", self._build_effect_row(self.threshold_amount, self.threshold_band))
        form.addRow("Warp Amount", self._build_effect_row(self.warp_amount, self.warp_band))
        form.addRow("Warp Speed", self._build_effect_row(self.warp_speed_amount, self.warp_speed_band))
        form.addRow("Rotation Mod", self._build_effect_row(self.rotation_mod_amount, self.rotation_mod_band))
        return form

    def _build_color_layout(self) -> QFormLayout:
        form = QFormLayout()
        form.addRow("Color", self._build_color_row())
        return form

    def _build_framed_section(self, title: str, content_layout: QFormLayout) -> QGroupBox:
        box = QGroupBox(title)
        box.setLayout(content_layout)
        return box

    def _build_color_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.color_button)
        layout.addWidget(self.hex_input)
        layout.addWidget(QLabel("R"))
        layout.addWidget(self.r_spin)
        layout.addWidget(QLabel("G"))
        layout.addWidget(self.g_spin)
        layout.addWidget(QLabel("B"))
        layout.addWidget(self.b_spin)
        layout.addStretch(1)
        return row

    def _build_smoothing_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.smoothing_check)
        layout.addWidget(self.smoothing_amount)
        layout.addStretch(1)
        return row

    def _make_effect_row(self, minimum: float, maximum: float, value: float):
        amount = QDoubleSpinBox()
        amount.setRange(minimum, maximum)
        amount.setSingleStep(max(0.01, (maximum - minimum) / 50))
        amount.setValue(value)
        band = QComboBox()
        self.effect_controls.append(band)
        return amount, band

    def _build_effect_row(self, amount: QDoubleSpinBox, band: QComboBox) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(amount)
        layout.addWidget(band)
        layout.addStretch(1)
        return row

    def _with_browse(self, line_edit: QLineEdit, handler) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        button = QPushButton("Browse")
        button.clicked.connect(handler)
        layout.addWidget(button)
        return row

    def _build_media_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.media_path)
        self.media_browse_button = QPushButton("Browse")
        self.media_browse_button.clicked.connect(self._browse_media)
        layout.addWidget(self.media_browse_button)
        return row

    def _build_transport_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self.play_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.loop_check)
        row.addWidget(self.mute_button)
        row.addStretch(1)
        return row

    def _build_in_out_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self.set_in_button)
        row.addWidget(self.preview_in_label)
        row.addSpacing(12)
        row.addWidget(self.set_out_button)
        row.addWidget(self.preview_out_label)
        row.addStretch(1)
        return row

    def _browse_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Media",
            filter="Media Files (*.gif *.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.mov *.mkv *.avi *.webm)",
        )
        if path:
            self.media_path.setText(path)

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select MP3", filter="Audio Files (*.mp3 *.wav)")
        if path:
            self.audio_path.setText(path)
            self._load_audio_preview(path)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output MP4", filter="MP4 Files (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_path.setText(path)

    def _on_render(self) -> None:
        settings = RenderSettings(
            media_path=self.media_path.text().strip(),
            audio_path=self.audio_path.text().strip(),
            output_path=self.output_path.text().strip(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            fps=self.fps_spin.value(),
            enable_line=self.line_check.isChecked(),
            enable_lissajous=self.lissajous_check.isChecked(),
            enable_glow=self.glow_check.isChecked(),
            bands=self.band_count,
            preserve_aspect=self.preserve_check.isChecked(),
            edge_mode=self.render_mode.currentData(),
            edge_method=self.edge_method.currentData(),
            edge_threshold=float(self.edge_threshold.value()),
            glow_strength=float(self.glow_strength.value()),
            color_rgb=(self.r_spin.value(), self.g_spin.value(), self.b_spin.value()),
            smoothing_enabled=self.smoothing_check.isChecked(),
            smoothing_amount=float(self.smoothing_amount.value()),
            mod_displace_x_amount=float(self.displace_x_amount.value()),
            mod_displace_x_band=self.displace_x_band.currentData(),
            mod_displace_y_amount=float(self.displace_y_amount.value()),
            mod_displace_y_band=self.displace_y_band.currentData(),
            mod_thickness_amount=float(self.thickness_amount.value()),
            mod_thickness_band=self.thickness_band.currentData(),
            mod_glow_amount=float(self.glow_amount.value()),
            mod_glow_band=self.glow_band.currentData(),
            mod_threshold_amount=float(self.threshold_amount.value()),
            mod_threshold_band=self.threshold_band.currentData(),
            mod_warp_amount=float(self.warp_amount.value()),
            mod_warp_band=self.warp_band.currentData(),
            mod_warp_speed_amount=float(self.warp_speed_amount.value()),
            mod_warp_speed_band=self.warp_speed_band.currentData(),
            media_mode=self.media_mode.currentData(),
            shape_type=self.shape_type.currentData(),
            polygon_sides=self.polygon_sides.value(),
            shape_rotation=float(self.shape_rotation.value()),
            mod_rotation_amount=float(self.rotation_mod_amount.value()),
            mod_rotation_band=self.rotation_mod_band.currentData(),
        )

        missing = []
        for name, value in asdict(settings).items():
            if not name.endswith("_path"):
                continue
            if name == "media_path" and settings.media_mode == "shapes":
                continue
            if not value:
                missing.append(name)
        if missing:
            QMessageBox.warning(self, "Missing paths", f"Missing: {', '.join(missing)}")
            return

        if settings.media_mode == "media":
            if not os.path.exists(settings.media_path):
                QMessageBox.warning(self, "Missing media", "Media file not found")
                return
        if not os.path.exists(settings.audio_path):
            QMessageBox.warning(self, "Missing audio", "Audio file not found")
            return

        self.render_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.render_stack.setCurrentIndex(1)

        self.worker = RenderWorker(settings)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress.setValue(0)
            return
        pct = int((current / total) * 100)
        self.progress.setValue(pct)

    def _on_failed(self, message: str) -> None:
        self.render_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.render_stack.setCurrentIndex(0)
        self.progress.setValue(0)
        QMessageBox.critical(self, "Render failed", message.strip() or "Unknown error")

    def _on_cancelled(self) -> None:
        self.render_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.render_stack.setCurrentIndex(0)
        self.progress.setValue(0)
        QMessageBox.information(self, "Cancelled", "Render cancelled")

    def _on_finished(self) -> None:
        self.render_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.render_stack.setCurrentIndex(0)
        self.progress.setValue(100)
        QMessageBox.information(self, "Done", "Render complete")

    def _on_cancel(self) -> None:
        if self.worker:
            self.cancel_button.setEnabled(False)
            self.worker.cancel()

    def _schedule_audio_reload(self) -> None:
        path = self.audio_path.text().strip()
        if path and os.path.exists(path):
            self._load_audio_preview(path)

    def _on_audio_path_committed(self) -> None:
        path = self.audio_path.text().strip()
        if path:
            self._load_audio_preview(path)

    def _load_audio_preview(self, path: str) -> None:
        if not path or not os.path.exists(path):
            self._set_preview_enabled(False)
            return

        self._audio_load_token += 1
        token = self._audio_load_token
        if self.audio_worker and self.audio_worker.isRunning():
            self.audio_worker.requestInterruption()
        self.audio_worker = AudioAnalysisWorker(
            path,
            fps=self.fps_spin.value(),
            bands=self.band_count,
        )
        self.audio_worker.finished.connect(lambda analysis: self._on_audio_loaded(token, analysis))
        self.audio_worker.failed.connect(self._on_audio_failed)
        self.audio_worker.start()

        self.preview_player.setSource(QUrl.fromLocalFile(path))
        self.preview_player.stop()

    def _on_audio_loaded(self, token: int, analysis: AudioAnalysis) -> None:
        if token != self._audio_load_token:
            return
        self.audio_analysis = analysis
        duration_ms = int(analysis.duration * 1000)
        self.preview_slider.setRange(0, max(0, duration_ms))
        self._preview_loop_in = 0
        self._preview_loop_out = duration_ms
        self._update_preview_labels(0, duration_ms)
        self.preview_slider.set_loop_region(self._preview_loop_in, self._preview_loop_out)
        self._preview_smoothed = None
        self._preview_phase = 0.0
        self._set_preview_enabled(True)
        self._tick_preview()

    def _on_audio_failed(self, message: str) -> None:
        self.audio_analysis = None
        self._set_preview_enabled(False)
        QMessageBox.warning(self, "Audio analysis failed", message.strip() or "Unknown error")

    def _set_preview_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.loop_check.setEnabled(enabled)
        self.mute_button.setEnabled(enabled)
        self.preview_slider.setEnabled(enabled)
        self.set_in_button.setEnabled(enabled)
        self.set_out_button.setEnabled(enabled)
        if not enabled:
            self.preview_slider.setValue(0)
            self.preview_slider.set_loop_region(0, 0)
            self.preview_time.setText("0:00 / 0:00")
            self.preview_in_label.setText("In: 0:00")
            self.preview_out_label.setText("Out: 0:00")

    def _toggle_preview_play(self) -> None:
        if self.preview_player.playbackState() == QMediaPlayer.PlayingState:
            self.preview_player.pause()
            self.preview_timer.stop()
            self._set_media_icons(is_playing=False)
            return
        if self.audio_analysis is None:
            return
        self.preview_player.play()
        self.preview_timer.start()
        self._set_media_icons(is_playing=True)

    def _stop_preview(self) -> None:
        self.preview_player.stop()
        self.preview_timer.stop()
        self._set_media_icons(is_playing=False)
        self.preview_slider.setValue(self._preview_loop_in)

    def _on_preview_duration(self, duration_ms: int) -> None:
        if duration_ms > 0:
            self.preview_slider.setRange(0, duration_ms)
            self._preview_loop_out = duration_ms
            self._update_preview_labels(self._preview_loop_in, self._preview_loop_out)
            self.preview_slider.set_loop_region(self._preview_loop_in, self._preview_loop_out)

    def _on_preview_position(self, position_ms: int) -> None:
        if not self._preview_dragging:
            self.preview_slider.setValue(position_ms)
        self._update_preview_time(position_ms)
        if self.loop_check.isChecked() and self._preview_loop_out > self._preview_loop_in:
            if position_ms >= self._preview_loop_out:
                self.preview_player.setPosition(self._preview_loop_in)

    def _on_preview_seek_start(self) -> None:
        self._preview_dragging = True

    def _on_preview_seek_end(self) -> None:
        self._preview_dragging = False
        self.preview_player.setPosition(self.preview_slider.value())
        self._preview_smoothed = None
        self._tick_preview()

    def _on_preview_seek(self, value: int) -> None:
        if self._preview_dragging:
            self._update_preview_time(value)
            self._tick_preview()

    def _set_preview_in(self) -> None:
        self._preview_loop_in = self.preview_slider.value()
        if self._preview_loop_out < self._preview_loop_in:
            self._preview_loop_out = self._preview_loop_in
        self._update_preview_labels(self._preview_loop_in, self._preview_loop_out)
        self.preview_slider.set_loop_region(self._preview_loop_in, self._preview_loop_out)

    def _set_preview_out(self) -> None:
        self._preview_loop_out = self.preview_slider.value()
        if self._preview_loop_out < self._preview_loop_in:
            self._preview_loop_in = self._preview_loop_out
        self._update_preview_labels(self._preview_loop_in, self._preview_loop_out)
        self.preview_slider.set_loop_region(self._preview_loop_in, self._preview_loop_out)

    def _update_preview_time(self, position_ms: int) -> None:
        total_ms = self.preview_slider.maximum()
        self.preview_time.setText(
            f"{self._format_time(position_ms)} / {self._format_time(total_ms)}"
        )

    def _update_preview_labels(self, in_ms: int, out_ms: int) -> None:
        self.preview_in_label.setText(f"In: {self._format_time(in_ms)}")
        self.preview_out_label.setText(f"Out: {self._format_time(out_ms)}")
        self._update_preview_time(self.preview_slider.value())

    def _toggle_mute(self, muted: bool) -> None:
        self.preview_audio.setMuted(muted)
        self._set_media_icons(is_playing=self.preview_player.playbackState() == QMediaPlayer.PlayingState)

    def _on_loop_changed(self, loop_in: int, loop_out: int) -> None:
        self._preview_loop_in = loop_in
        self._preview_loop_out = loop_out
        self._update_preview_labels(loop_in, loop_out)

    def _set_media_icons(self, *, is_playing: bool) -> None:
        play_icon = self.style().standardIcon(
            QStyle.SP_MediaPause if is_playing else QStyle.SP_MediaPlay
        )
        stop_icon = self.style().standardIcon(QStyle.SP_MediaStop)
        loop_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        in_icon = self.style().standardIcon(QStyle.SP_ArrowLeft)
        out_icon = self.style().standardIcon(QStyle.SP_ArrowRight)
        mute_icon = self.style().standardIcon(
            QStyle.SP_MediaVolumeMuted if self.mute_button.isChecked() else QStyle.SP_MediaVolume
        )
        self.play_button.setIcon(play_icon)
        self.stop_button.setIcon(stop_icon)
        self.loop_check.setIcon(loop_icon)
        self.mute_button.setIcon(mute_icon)
        self.set_in_button.setIcon(in_icon)
        self.set_out_button.setIcon(out_icon)

    def _format_time(self, ms: int) -> str:
        seconds = max(0, int(ms // 1000))
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"

    def _tick_preview(self) -> None:
        if self.audio_analysis is None:
            return
        position_ms = self.preview_slider.value()
        if self.preview_player.playbackState() == QMediaPlayer.PlayingState:
            position_ms = self.preview_player.position()
        time_sec = position_ms / 1000.0
        fps = self.fps_spin.value()
        frame_idx = int(time_sec * fps)
        band_values = band_at_frame(self.audio_analysis.band_energies, frame_idx)

        if self.smoothing_check.isChecked():
            if self._preview_smoothed is None:
                self._preview_smoothed = band_values.astype(float)
            else:
                alpha = float(max(0.0, min(1.0, self.smoothing_amount.value())))
                self._preview_smoothed = self._preview_smoothed + (band_values - self._preview_smoothed) * alpha
            band_values = self._preview_smoothed
        else:
            self._preview_smoothed = None

        energy = float(band_values.mean()) if band_values.size else 0.0
        displace_x = self.displace_x_amount.value() * self._select_band_value(
            band_values, self.displace_x_band.currentData()
        )
        displace_y = self.displace_y_amount.value() * self._select_band_value(
            band_values, self.displace_y_band.currentData()
        )
        thickness = 1.0 + self.thickness_amount.value() * self._select_band_value(
            band_values, self.thickness_band.currentData()
        )
        warp_amount = self.warp_amount.value() * self._select_band_value(
            band_values, self.warp_band.currentData()
        )
        warp_speed = 1.0 + self.warp_speed_amount.value() * self._select_band_value(
            band_values, self.warp_speed_band.currentData()
        )
        rotation_mod = self.rotation_mod_amount.value() * self._select_band_value(
            band_values, self.rotation_mod_band.currentData()
        )

        if self.preview_player.playbackState() == QMediaPlayer.PlayingState:
            self._preview_phase += (1.0 / max(1, fps)) * warp_speed * 6.0

        color = QColor(self.r_spin.value(), self.g_spin.value(), self.b_spin.value())
        preview_shape = "ring"
        preview_sides = self.polygon_sides.value()
        preview_rotation = float(self.shape_rotation.value())
        if self.media_mode.currentData() == "shapes":
            preview_shape = self.shape_type.currentData()
            preview_sides = self.polygon_sides.value()
            preview_rotation = float(self.shape_rotation.value() + rotation_mod)
        self.preview.update_state(
            color=color,
            band_values=band_values,
            energy=energy,
            displace_x=displace_x,
            displace_y=displace_y,
            thickness=thickness,
            warp_amount=warp_amount * 4.0,
            warp_phase=self._preview_phase,
            shape_type=preview_shape,
            polygon_sides=preview_sides,
            shape_rotation=preview_rotation,
        )

    def _select_band_value(self, bands: np.ndarray, selector: str) -> float:
        if bands.size == 0:
            return 0.0
        if selector == "all":
            return float(bands.mean())
        if selector == "low":
            return float(bands[0])
        if selector == "mid":
            return float(bands[bands.size // 2])
        if selector == "high":
            return float(bands[-1])
        if selector and selector.startswith("band:"):
            try:
                index = int(selector.split(":", 1)[1])
            except ValueError:
                return float(bands.mean())
            index = max(0, min(bands.size - 1, index))
            return float(bands[index])
        return float(bands.mean())

    def _wire_preview_controls(self) -> None:
        controls = [
            self.displace_x_amount,
            self.displace_y_amount,
            self.thickness_amount,
            self.glow_amount,
            self.threshold_amount,
            self.warp_amount,
            self.warp_speed_amount,
            self.rotation_mod_amount,
            self.shape_rotation,
            self.smoothing_check,
            self.smoothing_amount,
            self.r_spin,
            self.g_spin,
            self.b_spin,
        ]
        for control in controls:
            if hasattr(control, "valueChanged"):
                control.valueChanged.connect(self._tick_preview)
            if hasattr(control, "toggled"):
                control.toggled.connect(self._tick_preview)
        for combo in (
            self.displace_x_band,
            self.displace_y_band,
            self.thickness_band,
            self.glow_band,
            self.threshold_band,
            self.warp_band,
            self.warp_speed_band,
            self.rotation_mod_band,
        ):
            combo.currentIndexChanged.connect(self._tick_preview)

    def _on_mode_changed(self) -> None:
        edge_only = self.render_mode.currentData() == "edge_only"
        self.line_check.setEnabled(not edge_only)
        self.lissajous_check.setEnabled(not edge_only)

    def _on_media_mode_changed(self) -> None:
        shapes = self.media_mode.currentData() == "shapes"
        self.media_path.setEnabled(not shapes)
        if hasattr(self, "media_browse_button"):
            self.media_browse_button.setEnabled(not shapes)
        self.shape_type.setEnabled(shapes)
        self.shape_rotation.setEnabled(shapes)
        self.rotation_mod_amount.setEnabled(shapes)
        self.rotation_mod_band.setEnabled(shapes)
        self.polygon_sides.setEnabled(shapes and self.shape_type.currentData() == "polygon")

    def _on_shape_type_changed(self) -> None:
        self.polygon_sides.setEnabled(
            self.media_mode.currentData() == "shapes"
            and self.shape_type.currentData() == "polygon"
        )

    def _apply_theme(self) -> None:
        accent = "#00FBCC"
        self.setStyleSheet(
            f"""
            QWidget {{ background-color: #101010; color: #E8E8E8; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: #1C1C1C;
                border: 1px solid #2A2A2A;
                padding: 4px;
            }}
            QLineEdit::selection, QTextEdit::selection, QPlainTextEdit::selection {{
                background: {accent};
                color: #0B0B0B;
            }}
            QComboBox::drop-down {{ border-left: 1px solid #2A2A2A; }}
            QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
                border-right: 4px solid transparent; border-top: 6px solid {accent}; width: 0; height: 0; }}
            QAbstractItemView {{
                background: #1C1C1C;
                selection-background-color: {accent};
                selection-color: #0B0B0B;
                outline: none;
            }}
            QAbstractItemView::item:hover {{
                background: rgba(0, 251, 204, 80);
                color: #0B0B0B;
            }}
            QAbstractItemView::item:selected {{
                background: {accent};
                color: #0B0B0B;
            }}
            QPushButton, QToolButton {{
                background-color: #1C1C1C;
                border: 1px solid #2A2A2A;
                padding: 6px 10px;
            }}
            QPushButton:hover, QToolButton:hover {{ border-color: {accent}; }}
            QPushButton:checked, QToolButton:checked {{ border-color: {accent}; color: {accent}; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px; border: 1px solid #2A2A2A; background: #141414;
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
                border: 1px solid {accent};
            }}
            QSlider::groove:horizontal {{
                height: 6px; background: #1A1A1A; border: 1px solid #2A2A2A; border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 3px; }}
            QSlider::add-page:horizontal {{ background: #1A1A1A; }}
            QSlider::handle:horizontal {{
                width: 12px; margin: -4px 0; border-radius: 6px; background: {accent};
            }}
            QProgressBar {{
                border: 1px solid #2A2A2A; background: #141414; text-align: center;
            }}
            QProgressBar::chunk {{ background: {accent}; }}
            QScrollBar:vertical {{
                background: #141414; width: 10px; margin: 0; border: 1px solid #2A2A2A;
            }}
            QScrollBar::handle:vertical {{ background: {accent}; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none; border: none; height: 0;
            }}
            QMessageBox {{
                background-color: #101010;
            }}
            QMessageBox QLabel {{
                color: #E8E8E8;
            }}
            QMessageBox QPushButton {{
                border: 1px solid #2A2A2A;
            }}
            QGroupBox {{
                border: 1px solid #2A2A2A;
                margin-top: 10px;
                padding-top: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: {accent};
            }}
            """
        )


    def _open_color_dialog(self) -> None:
        current = QColor(self.r_spin.value(), self.g_spin.value(), self.b_spin.value())
        color = QColorDialog.getColor(current, self, "Pick render color")
        if color.isValid():
            self._set_color(color)

    def _set_color(self, color: QColor) -> None:
        if self._color_updating:
            return
        self._color_updating = True
        self.r_spin.setValue(color.red())
        self.g_spin.setValue(color.green())
        self.b_spin.setValue(color.blue())
        hex_value = f"#{color.red():02X}{color.green():02X}{color.blue():02X}"
        self.hex_input.setText(hex_value)
        self.color_button.setStyleSheet(f"background-color: {hex_value};")
        self._color_updating = False

    def _sync_from_hex(self) -> None:
        if self._color_updating:
            return
        text = self.hex_input.text().strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) != 6:
            return
        try:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
        except ValueError:
            return
        self._set_color(QColor(r, g, b))

    def _sync_from_rgb(self) -> None:
        if self._color_updating:
            return
        color = QColor(self.r_spin.value(), self.g_spin.value(), self.b_spin.value())
        self._set_color(color)

    def _update_band_selectors(self) -> None:
        options = [
            ("All", "all"),
            ("Subs", "band:0"),
            ("Lows", "band:1"),
            ("Low Mids", "band:2"),
            ("High Mids", "band:3"),
            ("Highs", "band:4"),
        ]
        for selector in self.effect_controls:
            current = selector.currentData()
            selector.blockSignals(True)
            selector.clear()
            for label, data in options:
                selector.addItem(label, data)
            data_values = [data for _, data in options]
            if current is not None and current in data_values:
                selector.setCurrentIndex(data_values.index(current))
            selector.blockSignals(False)

    def _set_band_defaults(self) -> None:
        defaults = {
            self.displace_x_band: "band:1",
            self.displace_y_band: "band:2",
            self.thickness_band: "all",
            self.glow_band: "all",
            self.threshold_band: "all",
            self.warp_band: "band:4",
            self.warp_speed_band: "band:3",
            self.rotation_mod_band: "all",
        }
        for combo, value in defaults.items():
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)


def main() -> int:
    app = QApplication([])
    window = MainWindow()
    window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
