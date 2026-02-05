from __future__ import annotations

import json
import os
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl, Signal, QSize, QIODevice, QThread, QPointF, QRect
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPixmap, QPolygonF, QPen
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QStyle,
    QStyleOptionSlider,
)

from .audio import AudioAnalysis, band_at_frame, load_and_analyze
from .render import (
    RenderCancelled,
    RenderSettings,
    render_video,
    _apply_hue_shift,
    _mod_value,
    _oscillator_value,
    _script_audio_payload,
    _rotation_direction,
    _text_to_polylines,
    _apply_bloom,
    _apply_vignette,
    _apply_phosphor_mask,
    _apply_chromatic_aberration,
    _apply_barrel_distortion,
    _apply_noise,
    _apply_horizontal_jitter,
    _apply_vertical_roll,
    _apply_color_bleed,
    _apply_dither,
)


ACCENT = "#00FBCC"


class PreviewCanvas(QWidget):
    resized = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._last_image: QImage | None = None
        self._polylines: list[list[tuple[float, float]]] | None = None
        self._shape_points: list[tuple[float, float]] | None = None
        self._color = QColor(0, 255, 204)
        self._energy = 0.0
        self._displace = (0.0, 0.0)
        self._thickness = 2.0
        self._warp_amount = 0.0
        self._warp_phase = 0.0
        self._shape_type = "ring"
        self._polygon_sides = 5
        self._shape_rotation = 0.0
        self._shape_size = 1.0
        self._shape_params: dict[str, float] = {}
        self._preserve_aspect = True
        self._glow_strength = 0.0
        self._glow_radius = 2.0
        self._trail_strength = 0.0
        self._flicker_amount = 0.0
        self._hue_shift = 0.0
        self._scanline_amount = 0.0
        self._scanline_speed = 1.0
        self._decimate_step = 1
        self._jitter_amount = 0.0
        self._threshold = 0.0
        self._time = 0.0
        self._dither_amount = 0.0
        self._phosphor_amount = 0.0
        self._bloom_amount = 0.0
        self._bloom_radius = 2.0
        self._bloom_threshold = 0.6
        self._vignette_amount = 0.0
        self._vignette_power = 1.8
        self._chroma_shift_x = 0.0
        self._chroma_shift_y = 0.0
        self._barrel_amount = 0.0
        self._noise_amount = 0.0
        self._h_jitter_amount = 0.0
        self._h_jitter_speed = 1.5
        self._v_roll_amount = 0.0
        self._v_roll_speed = 0.25
        self._bleed_amount = 0.0
        self.setMinimumSize(640, 360)

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self._shape_points = None
        self._polylines = None
        self._last_image = None
        self.update()

    def update_state(
        self,
        *,
        color: QColor,
        energy: float,
        displace_x: float,
        displace_y: float,
        thickness: float,
        warp_amount: float,
        warp_phase: float,
        shape_type: str,
        polygon_sides: int,
        shape_rotation: float,
        shape_size: float,
        shape_params: dict[str, float],
        polylines: list[list[tuple[float, float]]] | None,
        preserve_aspect: bool,
        glow_strength: float,
    ) -> None:
        self._image = None
        self._color = QColor(color)
        self._energy = float(energy)
        self._displace = (float(displace_x), float(displace_y))
        self._thickness = float(thickness)
        self._warp_amount = float(warp_amount)
        self._warp_phase = float(warp_phase)
        self._shape_type = shape_type
        self._polygon_sides = int(polygon_sides)
        self._shape_rotation = float(shape_rotation)
        self._shape_size = float(shape_size)
        self._shape_params = dict(shape_params)
        self._polylines = polylines
        self._preserve_aspect = preserve_aspect
        self._glow_strength = float(glow_strength)
        self._glow_radius = float(shape_params.get("glow_radius", 2.0))
        self._trail_strength = float(shape_params.get("trail_strength", 0.0))
        self._flicker_amount = float(shape_params.get("flicker_amount", 0.0))
        self._hue_shift = float(shape_params.get("hue_shift_amount", 0.0))
        self._scanline_amount = float(shape_params.get("scanline_amount", 0.0))
        self._scanline_speed = float(shape_params.get("scanline_speed", 1.0))
        self._decimate_step = max(1, int(shape_params.get("decimate_step", 1)))
        self._jitter_amount = float(shape_params.get("jitter_amount", 0.0))
        self._threshold = float(shape_params.get("threshold", 0.0))
        self._time = float(shape_params.get("time", 0.0))
        self._dither_amount = float(shape_params.get("dither_amount", 0.0))
        self._phosphor_amount = float(shape_params.get("phosphor_amount", 0.0))
        self._bloom_amount = float(shape_params.get("bloom_amount", 0.0))
        self._bloom_radius = float(shape_params.get("bloom_radius", 2.0))
        self._bloom_threshold = float(shape_params.get("bloom_threshold", 0.6))
        self._vignette_amount = float(shape_params.get("vignette_amount", 0.0))
        self._vignette_power = float(shape_params.get("vignette_power", 1.8))
        self._chroma_shift_x = float(shape_params.get("chroma_shift_x", 0.0))
        self._chroma_shift_y = float(shape_params.get("chroma_shift_y", 0.0))
        self._barrel_amount = float(shape_params.get("barrel_amount", 0.0))
        self._noise_amount = float(shape_params.get("noise_amount", 0.0))
        self._h_jitter_amount = float(shape_params.get("h_jitter_amount", 0.0))
        self._h_jitter_speed = float(shape_params.get("h_jitter_speed", 1.5))
        self._v_roll_amount = float(shape_params.get("v_roll_amount", 0.0))
        self._v_roll_speed = float(shape_params.get("v_roll_speed", 0.25))
        self._bleed_amount = float(shape_params.get("bleed_amount", 0.0))
        self._shape_points = self._build_shape_points()
        self._image = self._render_image()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.resized.emit()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 12, 14))
        painter.setPen(QColor(30, 40, 44))
        step = 40
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)
        if self._image is not None:
            pix = QPixmap.fromImage(self._image)
            target = self.rect()
            scaled = pix.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (target.width() - scaled.width()) // 2
            y = (target.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            return

        painter.setPen(QColor(140, 150, 150))
        painter.drawText(self.rect(), Qt.AlignCenter, "No preview")

    def _draw_shape(self, painter: QPainter) -> None:
        points = self._shape_points
        if not points:
            return
        width = self.width()
        height = self.height()
        scale = min(width, height) * 0.45 * 0.82 * self._shape_size
        center_x = width * 0.5 + self._displace[0]
        center_y = height * 0.5 + self._displace[1]
        poly = QPolygonF()
        warp = max(0.0, self._warp_amount) * 0.003
        step = self._decimate_step
        for i, (x, y) in enumerate(points):
            if step > 1 and (i % step) != 0:
                continue
            if self._jitter_amount > 0.0:
                x += (np.random.rand() - 0.5) * 2.0 * self._jitter_amount
                y += (np.random.rand() - 0.5) * 2.0 * self._jitter_amount
            angle = np.arctan2(y, x)
            radius = np.sqrt(x * x + y * y)
            radius *= 1.0 + warp * np.sin(angle * 3.0 + self._warp_phase)
            px = np.cos(angle) * radius
            py = np.sin(angle) * radius
            poly.append(
                QPointF(center_x + px * scale, center_y + py * scale)
            )

        color = QColor(self._color)
        boost = 1.0
        if self._threshold > 0.0 and self._energy < self._threshold:
            boost *= 0.3
        color.setRed(min(255, int(color.red() * boost)))
        color.setGreen(min(255, int(color.green() * boost)))
        color.setBlue(min(255, int(color.blue() * boost)))

        if self._glow_strength > 0.0:
            glow = QColor(color)
            glow.setAlpha(int(120 * min(1.0, self._glow_strength)))
            glow_pen = QPen(glow)
            glow_pen.setWidthF(max(2.0, self._glow_radius * 2.0))
            painter.setPen(glow_pen)
            painter.drawPolyline(poly)

        pen = QPen(color)
        pen.setWidthF(max(2.0, self._thickness))
        painter.setPen(pen)
        painter.drawPolyline(poly)

    def _draw_polylines(self, painter: QPainter) -> None:
        width = self.width()
        height = self.height()
        if self._preserve_aspect:
            scale = min(width, height) * 0.45
            offset_x = width * 0.5 + self._displace[0]
            offset_y = height * 0.5 + self._displace[1]
        rotation = float(self._shape_params.get("poly_rotation", 0.0))
        if rotation:
            rot = np.deg2rad(rotation)
            cos_r = np.cos(rot)
            sin_r = np.sin(rot)
        warp = max(0.0, self._warp_amount) * 0.003
        color = QColor(self._color)
        boost = 1.0
        color.setRed(min(255, int(color.red() * boost)))
        color.setGreen(min(255, int(color.green() * boost)))
        color.setBlue(min(255, int(color.blue() * boost)))
        pen = QPen(color)
        pen.setWidthF(max(2.0, self._thickness))
        painter.setPen(pen)
        for line in self._polylines or []:
            if not line:
                continue
            poly = QPolygonF()
            step = self._decimate_step
            for i, (x, y) in enumerate(line):
                if step > 1 and (i % step) != 0:
                    continue
                if self._jitter_amount > 0.0:
                    x += (np.random.rand() - 0.5) * 2.0 * self._jitter_amount
                    y += (np.random.rand() - 0.5) * 2.0 * self._jitter_amount
                if warp > 0.0:
                    angle = np.arctan2(y, x)
                    radius = np.sqrt(x * x + y * y)
                    radius *= 1.0 + warp * np.sin(angle * 3.0 + self._warp_phase)
                    x = np.cos(angle) * radius
                    y = np.sin(angle) * radius
                if rotation:
                    xr = x * cos_r - y * sin_r
                    yr = x * sin_r + y * cos_r
                    x, y = xr, yr
                if self._preserve_aspect:
                    px = offset_x + x * scale
                    py = offset_y + y * scale
                else:
                    px = (x * 0.5 + 0.5) * (width - 1)
                    py = (y * 0.5 + 0.5) * (height - 1)
                poly.append(QPointF(px, py))
            painter.drawPolyline(poly)

    def _build_shape_points(self) -> list[tuple[float, float]]:
        rotation = np.deg2rad(self._shape_rotation)
        points: list[tuple[float, float]] = []
        if self._shape_type == "polygon":
            sides = max(3, int(self._polygon_sides))
            angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
            for t in angles:
                points.append((np.cos(t + rotation), np.sin(t + rotation)))
            points.append(points[0])
            return points
        if self._shape_type == "ellipse":
            steps = 360
            rx = float(self._shape_params.get("ellipse_x", 1.2))
            ry = float(self._shape_params.get("ellipse_y", 0.8))
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi
                x = np.cos(t) * rx
                y = np.sin(t) * ry
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "rectangle":
            w = float(self._shape_params.get("rect_width", 1.0))
            h = float(self._shape_params.get("rect_height", 0.6))
            corners = [
                (-w, -h),
                (w, -h),
                (w, h),
                (-w, h),
                (-w, -h),
            ]
            for x, y in corners:
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "star":
            spikes = max(3, int(self._shape_params.get("star_points", 5)))
            inner = float(self._shape_params.get("star_inner", 0.45))
            for i in range(spikes * 2 + 1):
                t = (i / (spikes * 2)) * 2 * np.pi + rotation
                r = 1.0 if i % 2 == 0 else inner
                points.append((np.cos(t) * r, np.sin(t) * r))
            return points
        if self._shape_type == "spiral":
            turns = float(self._shape_params.get("spiral_turns", 3.5))
            growth = float(self._shape_params.get("spiral_growth", 1.0))
            steps = 360
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi * turns
                r = (i / steps) * growth
                x = np.cos(t) * r
                y = np.sin(t) * r
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "lemniscate":
            steps = 360
            scale = float(self._shape_params.get("lemniscate_scale", 1.4))
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi
                denom = 1 + np.sin(t) ** 2
                x = (np.cos(t) / denom) * scale
                y = (np.sin(t) * np.cos(t) / denom) * scale
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "cardioid":
            steps = 360
            scale = float(self._shape_params.get("cardioid_scale", 1.0))
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi
                r = (1 - np.cos(t)) * scale
                x = np.cos(t) * r
                y = np.sin(t) * r
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "clover":
            steps = 360
            petals = max(3, int(self._shape_params.get("clover_petals", 4)))
            scale = float(self._shape_params.get("clover_scale", 1.2))
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi
                r = np.cos(petals * t)
                x = r * np.cos(t) * scale
                y = r * np.sin(t) * scale
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "superellipse":
            steps = 360
            n = max(0.5, float(self._shape_params.get("superellipse_n", 2.6)))
            scale = float(self._shape_params.get("superellipse_scale", 1.1))
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi
                x = np.copysign(abs(np.cos(t)) ** (2 / n), np.cos(t)) * scale
                y = np.copysign(abs(np.sin(t)) ** (2 / n), np.sin(t)) * scale
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        if self._shape_type == "heart":
            steps = 360
            for i in range(steps + 1):
                t = (i / steps) * 2 * np.pi
                x = 16 * np.sin(t) ** 3 * 0.06
                y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) * 0.06
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                points.append((xr, yr))
            return points
        steps = 360
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi
            points.append((np.cos(t), np.sin(t)))
        return points

    def _render_image(self) -> QImage:
        image = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        image.fill(QColor(10, 12, 14))
        painter = QPainter(image)
        painter.setPen(QColor(30, 40, 44))
        step = 40
        for x in range(0, image.width(), step):
            painter.drawLine(x, 0, x, image.height())
        for y in range(0, image.height(), step):
            painter.drawLine(0, y, image.width(), y)

        if self._trail_strength > 0.0 and self._last_image is not None:
            painter.setOpacity(max(0.0, min(1.0, self._trail_strength)))
            painter.drawImage(0, 0, self._last_image)
            painter.setOpacity(1.0)

        if self._polylines:
            self._draw_polylines(painter)
        elif self._shape_points:
            self._draw_shape(painter)
        else:
            painter.setPen(QColor(140, 150, 150))
            painter.drawText(image.rect(), Qt.AlignCenter, "No preview")

        if self._scanline_amount > 0.0:
            painter.setPen(Qt.NoPen)
            amount = max(0.0, min(1.0, self._scanline_amount / 5.0))
            phase = self._time * self._scanline_speed * 6.0
            for y in range(0, image.height(), 3):
                wave = 0.5 + 0.5 * np.sin(y * 0.12 + phase)
                alpha = int(140 * amount * wave)
                if alpha <= 0:
                    continue
                painter.setBrush(QColor(0, 0, 0, alpha))
                painter.drawRect(0, y, image.width(), 1)

        if self._flicker_amount > 0.0:
            jitter = (np.random.rand() - 0.5) * 2.0
            factor = 1.0 + self._flicker_amount * (0.25 + 0.15 * jitter)
            factor = max(0.7, min(1.25, factor))
            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            delta = abs(factor - 1.0)
            alpha = int(min(1.0, delta / 0.3) * 120)
            if factor >= 1.0:
                overlay = QColor(255, 255, 255, alpha)
            else:
                overlay = QColor(0, 0, 0, alpha)
            painter.fillRect(image.rect(), overlay)
            painter.restore()

        painter.end()
        preview = image
        if (
            self._dither_amount > 0.0
            or self._phosphor_amount > 0.0
            or self._bloom_amount > 0.0
            or self._vignette_amount > 0.0
            or self._chroma_shift_x != 0.0
            or self._chroma_shift_y != 0.0
            or self._barrel_amount != 0.0
            or self._noise_amount > 0.0
            or self._h_jitter_amount > 0.0
            or self._v_roll_amount > 0.0
            or self._bleed_amount > 0.0
        ):
            rgb = preview.convertToFormat(QImage.Format_RGB888)
            width = rgb.width()
            height = rgb.height()
            buf = rgb.bits()
            buf.setsize(rgb.sizeInBytes())
            frame_rgb = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            grid_x = grid_x.astype(np.float32)
            grid_y = grid_y.astype(np.float32)

            frame_bgr = _apply_bloom(
                frame_bgr,
                amount=self._bloom_amount,
                radius=self._bloom_radius,
                threshold=self._bloom_threshold,
            )
            frame_bgr = _apply_vignette(
                frame_bgr,
                amount=self._vignette_amount,
                power=self._vignette_power,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            frame_bgr = _apply_phosphor_mask(frame_bgr, amount=self._phosphor_amount)
            frame_bgr = _apply_color_bleed(frame_bgr, amount=self._bleed_amount)
            frame_bgr = _apply_chromatic_aberration(
                frame_bgr,
                shift_x=self._chroma_shift_x,
                shift_y=self._chroma_shift_y,
            )
            if self._barrel_amount != 0.0:
                cx = (width - 1) / 2.0
                cy = (height - 1) / 2.0
                nx = (grid_x - cx) / max(1.0, cx)
                ny = (grid_y - cy) / max(1.0, cy)
                r2 = nx * nx + ny * ny
                factor = 1.0 + float(self._barrel_amount) * r2
                map_x = (nx * factor) * cx + cx
                map_y = (ny * factor) * cy + cy
                map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
                map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
                frame_bgr = _apply_barrel_distortion(frame_bgr, map_x, map_y)
            frame_bgr = _apply_horizontal_jitter(
                frame_bgr,
                amount=self._h_jitter_amount,
                speed=self._h_jitter_speed,
                t=self._time,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            frame_bgr = _apply_vertical_roll(
                frame_bgr,
                amount=self._v_roll_amount,
                speed=self._v_roll_speed,
                t=self._time,
            )
            frame_bgr = _apply_noise(frame_bgr, amount=self._noise_amount)
            frame_bgr = _apply_dither(frame_bgr, amount=self._dither_amount)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            preview = QImage(frame_rgb.data, width, height, frame_rgb.strides[0], QImage.Format_RGB888).copy()

        self._last_image = preview.copy()
        return preview


class LoopSlider(QSlider):
    loopChanged = Signal(int, int)

    def __init__(self) -> None:
        super().__init__(Qt.Horizontal)
        self.loop_enabled = False
        self.loop_in = 0
        self.loop_out = 0
        self._drag_mode: str | None = None

    def set_loop(self, loop_in: int, loop_out: int) -> None:
        self.loop_in = max(self.minimum(), int(loop_in))
        self.loop_out = max(self.loop_in, int(loop_out))
        self.update()
        self.loopChanged.emit(self.loop_in, self.loop_out)

    def set_loop_enabled(self, enabled: bool) -> None:
        self.loop_enabled = bool(enabled)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        if not self.loop_enabled or self.maximum() <= self.minimum():
            return
        painter = QPainter(self)
        opt = self.style().subControlRect(
            QStyle.CC_Slider,
            self._style_option(),
            QStyle.SC_SliderGroove,
            self,
        )
        span = self.maximum() - self.minimum()
        if span <= 0:
            return
        ratio_in = (self.loop_in - self.minimum()) / span
        ratio_out = (self.loop_out - self.minimum()) / span
        left = int(opt.left() + ratio_in * opt.width())
        right = int(opt.left() + ratio_out * opt.width())
        rect = opt.adjusted(0, -2, 0, 2)
        rect.setLeft(left)
        rect.setRight(max(left + 1, right))
        painter.setBrush(QColor(0, 251, 204, 60))
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)

        handle_color = QColor(0, 251, 204, 200)
        handle_width = 6
        painter.fillRect(left - handle_width // 2, opt.top(), handle_width, opt.height(), handle_color)
        painter.fillRect(right - handle_width // 2, opt.top(), handle_width, opt.height(), handle_color)

        playhead_x = self._value_to_pos(self.value(), opt)
        play_color = QColor(255, 160, 0, 220)
        painter.fillRect(playhead_x - 1, opt.top() - 4, 2, opt.height() + 8, play_color)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self.loop_enabled:
            opt = self.style().subControlRect(
                QStyle.CC_Slider,
                self._style_option(),
                QStyle.SC_SliderGroove,
                self,
            )
            x = event.position().x()
            left = self._value_to_pos(self.loop_in, opt)
            right = self._value_to_pos(self.loop_out, opt)
            if abs(x - left) <= 6:
                self._drag_mode = "in"
                return
            if abs(x - right) <= 6:
                self._drag_mode = "out"
                return
        self._drag_mode = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self.loop_enabled and self._drag_mode:
            opt = self.style().subControlRect(
                QStyle.CC_Slider,
                self._style_option(),
                QStyle.SC_SliderGroove,
                self,
            )
            value = self._pos_to_value(event.position().x(), opt)
            if self._drag_mode == "in":
                self.set_loop(min(value, self.loop_out), self.loop_out)
            elif self._drag_mode == "out":
                self.set_loop(self.loop_in, max(value, self.loop_in))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self._drag_mode = None
        super().mouseReleaseEvent(event)

    def _style_option(self) -> QStyleOptionSlider:
        opt = QStyleOptionSlider()
        opt.initFrom(self)
        opt.orientation = Qt.Horizontal
        opt.minimum = self.minimum()
        opt.maximum = self.maximum()
        opt.sliderPosition = self.value()
        opt.sliderValue = self.value()
        return opt

    def _value_to_pos(self, value: int, groove) -> int:  # noqa: ANN001
        span = max(1, self.maximum() - self.minimum())
        ratio = (value - self.minimum()) / span
        return groove.left() + int(groove.width() * ratio)

    def _pos_to_value(self, x: float, groove) -> int:  # noqa: ANN001
        span = max(1, groove.width())
        ratio = (x - groove.left()) / span
        value = self.minimum() + int((self.maximum() - self.minimum()) * ratio)
        return max(self.minimum(), min(self.maximum(), value))

class OscillatorAudioDevice(QIODevice):
    def __init__(self, waveform_cb: Callable[[], str], freq_cb: Callable[[], float]) -> None:
        super().__init__()
        self._waveform_cb = waveform_cb
        self._freq_cb = freq_cb
        self._phase = 0.0
        self._sample_rate = 44100

    def start(self) -> None:
        self.open(QIODevice.ReadOnly)

    def stop(self) -> None:
        self.close()

    def readData(self, maxlen: int) -> bytes:  # noqa: N802
        samples = max(1, maxlen // 2)
        freq = max(0.0, float(self._freq_cb()))
        wave = self._waveform_cb()
        t = (np.arange(samples) + 0.0) / float(self._sample_rate)
        phase = self._phase + 2 * np.pi * freq * t
        if wave == "triangle":
            raw = 2.0 * np.abs(2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        elif wave == "square":
            raw = np.where(np.sin(phase) >= 0, 1.0, -1.0)
        elif wave == "saw":
            raw = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
        else:
            raw = np.sin(phase)
        self._phase = (phase[-1] + (2 * np.pi * freq / self._sample_rate)) % (2 * np.pi)
        audio = (raw * 0.08 * 32767.0).astype(np.int16)
        return audio.tobytes()

    def writeData(self, data: bytes) -> int:  # noqa: N802
        return 0

    def bytesAvailable(self) -> int:  # noqa: N802
        return 4096


class RenderWorker(QThread):
    progress = Signal(int, int)
    finished = Signal()
    canceled = Signal()
    failed = Signal(str)

    def __init__(self, settings: RenderSettings) -> None:
        super().__init__()
        self._settings = settings
        self._cancel = False

    def run(self) -> None:  # noqa: D401
        try:
            render_video(
                self._settings,
                progress_cb=self.progress.emit,
                cancel_cb=self._cancel_check,
            )
        except RenderCancelled:
            self.canceled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        else:
            self.finished.emit()

    def cancel(self) -> None:
        self._cancel = True

    def _cancel_check(self) -> bool:
        return self._cancel


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
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(analysis)


class EffectWidget(QFrame):
    removed = Signal(str)

    def __init__(self, key: str, title: str, content: QWidget) -> None:
        super().__init__()
        self.key = key
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        header = QHBoxLayout()
        label = QLabel(title)
        remove = QToolButton()
        remove.setText("X")
        remove.setToolTip("Remove effect")
        remove.setStyleSheet("color: #ff5a5a; font-weight: bold;")
        remove.clicked.connect(lambda: self.removed.emit(self.key))
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(remove)
        layout.addLayout(header)
        layout.addWidget(content)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.setWindowTitle("Oscimorph")
        self._set_branding()
        self._apply_theme()

        self.preview_canvas = PreviewCanvas()
        self.preview_canvas.resized.connect(self._update_preview_buffer)

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        self.preview_timer.timeout.connect(self._tick_preview)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self._sync_slider_from_media)
        self.media_player.durationChanged.connect(self._sync_duration_from_media)

        self.osc_audio_sink: QAudioSink | None = None
        self.osc_audio_device: OscillatorAudioDevice | None = None

        self.audio_analysis: AudioAnalysis | None = None
        self.audio_worker: AudioAnalysisWorker | None = None
        self.script_generate: Callable | None = None

        self.preview_time = 0.0
        self.preview_fps = 30
        self.preview_smoothed: np.ndarray | None = None

        self.loop_in_ms = 0
        self.loop_out_ms = 0
        self._preview_dragging = False

        self._build_ui()
        self._wire_events()
        self._update_preview_buffer()
        self._refresh_effects_dropdown()
        self._update_visibility()
        self._update_preview_labels(self.loop_in_ms, self.loop_out_ms)
        self._update_preview_frame()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: #0c0f12;
                color: #d7e7e7;
                font-family: \"Segoe UI\";
                font-size: 12px;
            }}
            QGroupBox {{
                border: 1px solid #1c262b;
                border-radius: 6px;
                margin-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {ACCENT};
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background: #0f1418;
                border: 1px solid #243238;
                padding: 4px;
                border-radius: 4px;
                selection-background-color: {ACCENT};
                selection-color: #0c0f12;
            }}
            QComboBox QAbstractItemView {{
                selection-background-color: {ACCENT};
                selection-color: #0c0f12;
                background: #0f1418;
                border: 1px solid #243238;
            }}
            QPushButton, QToolButton {{
                background-color: #11161a;
                border: 1px solid #243238;
                padding: 4px 8px;
                border-radius: 4px;
            }}
            QPushButton:hover, QToolButton:hover {{
                border-color: {ACCENT};
            }}
            QPushButton:checked, QToolButton:checked {{
                background-color: #0f2b29;
                border-color: {ACCENT};
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #1a2328;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 12px;
                margin: -6px 0;
                border-radius: 6px;
            }}
            QProgressBar {{
                border: 1px solid #243238;
                background: #11161a;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT};
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 1px solid #243238;
                border-radius: 3px;
                background: #0f1418;
            }}
            QCheckBox::indicator:checked {{
                background: {ACCENT};
                border-color: {ACCENT};
            }}
            """
        )

    def _set_branding(self) -> None:
        icon_path = os.path.join(self._app_root, "assets", "logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)

        preview_header = QHBoxLayout()
        preview_title = QLabel("Preview")
        preview_title.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        logo_label = QLabel()
        logo_path = os.path.join(self._app_root, "assets", "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                logo_label.setPixmap(pixmap.scaledToHeight(28, Qt.SmoothTransformation))
        preview_header.addWidget(logo_label)
        preview_info = QLabel("Preview (proxy)\nMedia cannot be previewed")
        preview_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(preview_info)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.preview_canvas, 1)

        transport = QHBoxLayout()
        self.play_button = self._tool_button(QStyle.SP_MediaPlay, "Play")
        self.play_button.setCheckable(True)
        self.stop_button = self._tool_button(QStyle.SP_MediaStop, "Stop")
        self.loop_button = self._tool_button(QStyle.SP_BrowserReload, "Loop")
        self.loop_button.setCheckable(True)
        self.set_in_button = self._tool_button(QStyle.SP_ArrowLeft, "Set In")
        self.set_out_button = self._tool_button(QStyle.SP_ArrowRight, "Set Out")
        self.mute_button = self._tool_button(QStyle.SP_MediaVolumeMuted, "Mute")
        self.mute_button.setCheckable(True)
        transport.addWidget(self.play_button)
        transport.addWidget(self.stop_button)
        transport.addWidget(self.loop_button)
        transport.addWidget(self.set_in_button)
        transport.addWidget(self.set_out_button)
        transport.addStretch(1)
        transport.addWidget(self.mute_button)
        preview_layout.addLayout(transport)

        self.timeline = LoopSlider()
        self.timeline.setRange(0, 1000)
        preview_layout.addWidget(self.timeline)

        self.preview_time_label = QLabel("0:00 / 0:00")
        preview_layout.addWidget(self.preview_time_label)

        in_out_row = QHBoxLayout()
        self.preview_in_label = QLabel("In: 0:00")
        self.preview_out_label = QLabel("Out: 0:00")
        in_out_row.addWidget(self.preview_in_label)
        in_out_row.addStretch(1)
        in_out_row.addWidget(self.preview_out_label)
        preview_layout.addLayout(in_out_row)

        splitter.addWidget(preview_panel)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        side_root = QWidget()
        self.side_layout = QVBoxLayout(side_root)

        self._build_io_frame()
        self._build_shape_frame()
        self._build_text_frame()
        self._build_script_frame()
        self._build_effects_frame()
        self._build_osc_frame()

        self.side_layout.addStretch(1)
        self.side_scroll.setWidget(side_root)
        splitter.addWidget(self.side_scroll)
        splitter.setSizes([900, 900])

        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter)

    def _build_io_frame(self) -> None:
        self.io_frame = QGroupBox("Inputs & Output")
        content = self._collapsible_content(self.io_frame)
        form = QFormLayout(content)

        self.media_mode_combo = QComboBox()
        self.media_mode_combo.addItem("Media", "media")
        self.media_mode_combo.addItem("Shapes", "shapes")
        self.media_mode_combo.addItem("Text", "text")
        self.media_mode_combo.addItem("Script", "script")
        form.addRow("Input Mode", self.media_mode_combo)

        self.media_path = QLineEdit()
        self.media_browse = QPushButton("Browse")
        media_row = QHBoxLayout()
        media_row.addWidget(self.media_path, 1)
        media_row.addWidget(self.media_browse)
        form.addRow("Media Path", self._wrap_row(media_row))

        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItem("Audio File", "file")
        self.audio_mode_combo.addItem("Oscillator", "osc")
        form.addRow("Audio Source", self.audio_mode_combo)

        self.audio_path = QLineEdit()
        self.audio_browse = QPushButton("Browse")
        audio_row = QHBoxLayout()
        audio_row.addWidget(self.audio_path, 1)
        audio_row.addWidget(self.audio_browse)
        form.addRow("Audio Path", self._wrap_row(audio_row))

        self.output_path = QLineEdit(os.path.join("output", "output.mp4"))
        self.output_browse = QPushButton("Browse")
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_path, 1)
        out_row.addWidget(self.output_browse)
        form.addRow("Output Path", self._wrap_row(out_row))

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(1280)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(720)
        size_row = QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.height_spin)
        form.addRow("Resolution", self._wrap_row(size_row))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(30)
        form.addRow("FPS", self.fps_spin)

        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem("Edge Only", "edge_only")
        self.render_mode_combo.addItem("Edge Overlay", "edge_overlay")
        self.render_mode_combo.setCurrentIndex(0)
        form.addRow("Render Mode", self.render_mode_combo)

        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItem("Sobel", "sobel")
        self.edge_method_combo.addItem("Canny", "canny")
        form.addRow("Edge Method", self.edge_method_combo)

        self.edge_threshold = QDoubleSpinBox()
        self.edge_threshold.setRange(0.0, 0.9)
        self.edge_threshold.setSingleStep(0.01)
        self.edge_threshold.setValue(0.08)
        form.addRow("Edge Threshold", self.edge_threshold)

        self.preserve_aspect = QCheckBox("Preserve Aspect Ratio")
        self.preserve_aspect.setChecked(True)
        form.addRow(self.preserve_aspect)

        self.enable_line = QCheckBox("Waveform Line")
        self.enable_line.setChecked(True)
        self.enable_lissajous = QCheckBox("Lissajous")
        self.enable_lissajous.setChecked(True)
        line_row = QHBoxLayout()
        line_row.addWidget(self.enable_line)
        line_row.addWidget(self.enable_lissajous)
        form.addRow("Overlay", self._wrap_row(line_row))

        self.color_button = QPushButton("Pick Color")
        self.color_button.setStyleSheet(f"border-color: {ACCENT}; background-color: #00FFCC;")
        self.r_spin = QSpinBox()
        self.g_spin = QSpinBox()
        self.b_spin = QSpinBox()
        for spin in (self.r_spin, self.g_spin, self.b_spin):
            spin.setRange(0, 255)
            spin.setFixedWidth(60)
        self.r_spin.setValue(0)
        self.g_spin.setValue(255)
        self.b_spin.setValue(204)
        color_row = QHBoxLayout()
        color_row.addWidget(self.color_button)
        color_row.addWidget(self.r_spin)
        color_row.addWidget(self.g_spin)
        color_row.addWidget(self.b_spin)
        form.addRow("Color", self._wrap_row(color_row))

        self.glow_strength = QDoubleSpinBox()
        self.glow_strength.setRange(0.0, 3.0)
        self.glow_strength.setSingleStep(0.05)
        self.glow_strength.setValue(0.85)
        form.addRow("Glow Strength", self.glow_strength)

        self.render_button = QPushButton("Render")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.cancel_button = QToolButton()
        self.cancel_button.setText("X")
        self.cancel_button.setToolTip("Cancel render")
        self.cancel_button.setStyleSheet("color: #ff5a5a; font-weight: bold;")
        self.cancel_button.setVisible(False)
        render_row = QHBoxLayout()
        render_row.addWidget(self.render_button)
        render_row.addWidget(self.progress, 1)
        render_row.addWidget(self.cancel_button)
        form.addRow(self._wrap_row(render_row))

        self.side_layout.addWidget(self.io_frame)

    def _build_shape_frame(self) -> None:
        self.shape_frame = QGroupBox("Shapes")
        content = self._collapsible_content(self.shape_frame)
        layout = QVBoxLayout(content)

        row = QHBoxLayout()
        row.addWidget(QLabel("Shape"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Ring", "ring")
        self.shape_combo.addItem("Polygon", "polygon")
        self.shape_combo.addItem("Ellipse", "ellipse")
        self.shape_combo.addItem("Heart", "heart")
        self.shape_combo.addItem("Star", "star")
        self.shape_combo.addItem("Rectangle", "rectangle")
        self.shape_combo.addItem("Spiral", "spiral")
        self.shape_combo.addItem("Lemniscate", "lemniscate")
        self.shape_combo.addItem("Cardioid", "cardioid")
        self.shape_combo.addItem("Clover", "clover")
        self.shape_combo.addItem("Superellipse", "superellipse")
        row.addWidget(self.shape_combo, 1)
        layout.addLayout(row)

        common = QFormLayout()
        self.shape_size = QDoubleSpinBox()
        self.shape_size.setRange(0.1, 2.5)
        self.shape_size.setValue(1.0)
        common.addRow("Size", self.shape_size)
        self.shape_rotation = QDoubleSpinBox()
        self.shape_rotation.setRange(0.0, 360.0)
        self.shape_rotation.setValue(0.0)
        common.addRow("Orientation", self.shape_rotation)
        layout.addLayout(common)

        self.shape_params_frame = QGroupBox("Shape Params")
        self.shape_params_layout = QVBoxLayout(self.shape_params_frame)
        self.shape_param_widgets: dict[str, QWidget] = {}

        self._add_shape_param("polygon", "Sides", self._int_spin(3, 20, 5))
        self._add_shape_param("star", "Points", self._int_spin(3, 20, 5))
        self._add_shape_param("star", "Inner", self._float_spin(0.1, 0.95, 0.45))
        self._add_shape_param("rectangle", "Width", self._float_spin(0.2, 2.0, 1.0))
        self._add_shape_param("rectangle", "Height", self._float_spin(0.2, 2.0, 0.6))
        self._add_shape_param("ellipse", "X Radius", self._float_spin(0.4, 2.0, 1.2))
        self._add_shape_param("ellipse", "Y Radius", self._float_spin(0.4, 2.0, 0.8))
        self._add_shape_param("spiral", "Turns", self._float_spin(0.5, 8.0, 3.5))
        self._add_shape_param("spiral", "Growth", self._float_spin(0.2, 2.5, 1.0))
        self._add_shape_param("lemniscate", "Scale", self._float_spin(0.5, 2.5, 1.4))
        self._add_shape_param("cardioid", "Scale", self._float_spin(0.5, 2.0, 1.0))
        self._add_shape_param("clover", "Petals", self._int_spin(3, 12, 4))
        self._add_shape_param("clover", "Scale", self._float_spin(0.5, 2.0, 1.2))
        self._add_shape_param("superellipse", "N", self._float_spin(0.5, 6.0, 2.6))
        self._add_shape_param("superellipse", "Scale", self._float_spin(0.5, 2.0, 1.1))

        layout.addWidget(self.shape_params_frame)
        self.side_layout.addWidget(self.shape_frame)

    def _build_script_frame(self) -> None:
        self.script_frame = QGroupBox("Script")
        content = self._collapsible_content(self.script_frame)
        form = QFormLayout(content)

        self.script_path = QLineEdit()
        self.script_browse = QPushButton("Browse")
        script_row = QHBoxLayout()
        script_row.addWidget(self.script_path, 1)
        script_row.addWidget(self.script_browse)
        form.addRow("Script Path", self._wrap_row(script_row))
        self.side_layout.addWidget(self.script_frame)

    def _build_text_frame(self) -> None:
        self.text_frame = QGroupBox("Text")
        content = self._collapsible_content(self.text_frame)
        form = QFormLayout(content)

        self.text_input = QLineEdit("OSCIMORPH")
        form.addRow("Text", self.text_input)

        self.text_scale = QDoubleSpinBox()
        self.text_scale.setRange(0.2, 4.0)
        self.text_scale.setSingleStep(0.05)
        self.text_scale.setValue(1.0)
        form.addRow("Scale", self.text_scale)

        self.text_font_family = QFont().defaultFamily()
        self.text_font_combo = QComboBox()
        families = QFontDatabase.families()
        self.text_font_combo.addItems(families)
        if self.text_font_family in families:
            self.text_font_combo.setCurrentText(self.text_font_family)
        form.addRow("Font", self.text_font_combo)

        self.text_input.textChanged.connect(self._update_preview_frame)
        self.text_scale.valueChanged.connect(self._update_preview_frame)
        self.text_font_combo.currentTextChanged.connect(self._on_text_font_changed)

        self.side_layout.addWidget(self.text_frame)

    def _build_effects_frame(self) -> None:
        self.effects_frame = QGroupBox("Effects")
        content = self._collapsible_content(self.effects_frame)
        layout = QVBoxLayout(content)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets"))
        self.save_preset_button = QPushButton("Save")
        self.load_preset_button = QPushButton("Load")
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.load_preset_button)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Add Effect"))
        self.add_effect_combo = QComboBox()
        add_row.addWidget(self.add_effect_combo, 1)
        layout.addLayout(add_row)

        self.effects_container = QVBoxLayout()
        layout.addLayout(self.effects_container)
        self.side_layout.addWidget(self.effects_frame)

        self.effect_widgets: dict[str, EffectWidget] = {}
        self.effect_controls: dict[str, dict[str, QWidget]] = {}
        self.active_effects: set[str] = set()

        self.effect_groups = {
            "displace": ["mod_displace_x_amount", "mod_displace_y_amount"],
            "thickness": ["mod_thickness_amount"],
            "glow": ["mod_glow_amount", "glow_radius"],
            "threshold": ["mod_threshold_amount"],
            "warp": ["mod_warp_amount", "mod_warp_speed_amount"],
            "rotation": ["mod_rotation_amount"],
            "trail": ["trail_strength"],
            "flicker": ["flicker_amount"],
            "hue": ["hue_shift_amount"],
            "scanline": ["scanline_amount"],
            "decimate": ["decimate_step"],
            "jitter": ["jitter_amount"],
            "dither": ["dither_amount"],
            "phosphor": ["phosphor_amount"],
            "bloom": ["bloom_amount", "bloom_radius", "bloom_threshold"],
            "vignette": ["vignette_amount", "vignette_power"],
            "chromatic": ["chroma_shift_x", "chroma_shift_y"],
            "barrel": ["barrel_amount"],
            "noise": ["noise_amount"],
            "h_jitter": ["h_jitter_amount", "h_jitter_speed"],
            "v_roll": ["v_roll_amount", "v_roll_speed"],
            "bleed": ["bleed_amount"],
        }

        self.default_effects = {
            "smoothing",
            "displace",
            "thickness",
            "glow",
            "threshold",
            "warp",
            "rotation",
        }

        self._register_effects()

    def _build_osc_frame(self) -> None:
        self.osc_frame = QGroupBox("Oscillator")
        content = self._collapsible_content(self.osc_frame)
        form = QFormLayout(content)

        self.osc_waveform = QComboBox()
        self.osc_waveform.addItem("Sine", "sine")
        self.osc_waveform.addItem("Triangle", "triangle")
        self.osc_waveform.addItem("Square", "square")
        self.osc_waveform.addItem("Saw", "saw")
        form.addRow("Waveform", self.osc_waveform)

        self.osc_frequency = QDoubleSpinBox()
        self.osc_frequency.setRange(0.0, 20000.0)
        self.osc_frequency.setSingleStep(0.1)
        self.osc_frequency.setValue(0.5)
        form.addRow("Frequency (Hz)", self.osc_frequency)

        self.osc_depth = QDoubleSpinBox()
        self.osc_depth.setRange(0.0, 2.0)
        self.osc_depth.setSingleStep(0.05)
        self.osc_depth.setValue(1.0)
        form.addRow("Depth", self.osc_depth)

        self.osc_mix = QDoubleSpinBox()
        self.osc_mix.setRange(0.0, 1.0)
        self.osc_mix.setSingleStep(0.05)
        self.osc_mix.setValue(0.0)
        form.addRow("Osc Mix", self.osc_mix)

        self.osc_duration = QDoubleSpinBox()
        self.osc_duration.setRange(0.5, 600.0)
        self.osc_duration.setSingleStep(0.5)
        self.osc_duration.setValue(10.0)
        form.addRow("Duration (s)", self.osc_duration)

        self.osc_audio_monitor = QCheckBox("Oscillator Audio Monitor")
        form.addRow(self.osc_audio_monitor)

        self.side_layout.addWidget(self.osc_frame)

    def _wire_events(self) -> None:
        self.media_browse.clicked.connect(self._browse_media)
        self.audio_browse.clicked.connect(self._browse_audio)
        self.output_browse.clicked.connect(self._browse_output)
        self.script_browse.clicked.connect(self._browse_script)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.load_preset_button.clicked.connect(self._load_preset)
        self.media_mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.audio_mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.shape_combo.currentIndexChanged.connect(self._update_shape_params)
        self.script_path.textChanged.connect(self._reload_script)
        self.color_button.clicked.connect(self._pick_color)
        self.r_spin.valueChanged.connect(self._on_color_spin)
        self.g_spin.valueChanged.connect(self._on_color_spin)
        self.b_spin.valueChanged.connect(self._on_color_spin)
        self.render_button.clicked.connect(self._start_render)
        self.cancel_button.clicked.connect(self._cancel_render)

        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button.clicked.connect(self._stop_playback)
        self.loop_button.clicked.connect(self._toggle_loop)
        self.set_in_button.clicked.connect(self._set_loop_in)
        self.set_out_button.clicked.connect(self._set_loop_out)
        self.mute_button.clicked.connect(self._toggle_mute)
        self.timeline.sliderPressed.connect(self._on_preview_seek_start)
        self.timeline.sliderReleased.connect(self._on_preview_seek_end)
        self.timeline.valueChanged.connect(self._on_preview_seek)
        self.timeline.loopChanged.connect(self._on_loop_changed)

        for widget in self._preview_update_widgets():
            widget.valueChanged.connect(self._update_preview_frame)
        for widget in self._preview_update_checks():
            widget.toggled.connect(self._update_preview_frame)
        for widget in self._preview_update_combos():
            widget.currentIndexChanged.connect(self._update_preview_frame)

        self.add_effect_combo.currentIndexChanged.connect(self._add_effect_from_combo)
        self.osc_audio_monitor.toggled.connect(self._toggle_osc_audio)
        self.osc_duration.valueChanged.connect(self._sync_duration_for_osc)

    def _preview_update_widgets(self) -> list[QDoubleSpinBox | QSpinBox]:
        return [
            self.edge_threshold,
            self.glow_strength,
            self.width_spin,
            self.height_spin,
            self.fps_spin,
            self.shape_size,
            self.shape_rotation,
            self.osc_frequency,
            self.osc_depth,
            self.osc_mix,
            self.osc_duration,
            self.r_spin,
            self.g_spin,
            self.b_spin,
        ]

    def _preview_update_checks(self) -> list[QCheckBox]:
        return [
            self.preserve_aspect,
            self.enable_line,
            self.enable_lissajous,
        ]

    def _preview_update_combos(self) -> list[QComboBox]:
        return [
            self.media_mode_combo,
            self.shape_combo,
            self.edge_method_combo,
            self.render_mode_combo,
            self.osc_waveform,
        ]

    def _tool_button(self, icon_enum: QStyle.StandardPixmap, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(self.style().standardIcon(icon_enum))
        btn.setToolTip(tooltip)
        btn.setIconSize(QSize(18, 18))
        return btn

    def _collapsible_content(self, group: QGroupBox) -> QWidget:
        group.setCheckable(True)
        group.setChecked(True)
        wrapper = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 18, 8, 8)
        layout.addWidget(wrapper)
        group.toggled.connect(wrapper.setVisible)
        return wrapper

    def _wrap_row(self, layout: QHBoxLayout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper

    def _int_spin(self, low: int, high: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    def _float_spin(self, low: float, high: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    def _add_shape_param(self, shape: str, label: str, widget: QWidget) -> None:
        if shape not in self.shape_param_widgets:
            container = QWidget()
            form = QFormLayout(container)
            form.setContentsMargins(0, 0, 0, 0)
            self.shape_param_widgets[shape] = container
            self.shape_params_layout.addWidget(container)
        else:
            container = self.shape_param_widgets[shape]
            form = container.layout()
        form.addRow(label, widget)
        widget.valueChanged.connect(self._update_preview_frame)

    def _register_effects(self) -> None:
        self._register_effect("smoothing", "Smoothing", self._build_smoothing_effect())
        self._register_effect("displace", "Displace", self._build_displace_effect())
        self._register_effect(
            "thickness",
            "Thickness Mod",
            self._build_band_effect("mod_thickness_amount", 0.0, 10.0, 3.0, "all"),
        )
        self._register_effect("glow", "Glow", self._build_glow_combo())
        self._register_effect(
            "threshold",
            "Threshold Mod",
            self._build_band_effect("mod_threshold_amount", 0.0, 0.5, 0.05, "all"),
        )
        self._register_effect("warp", "Warp", self._build_warp_effect())
        self._register_effect("rotation", "Rotation", self._build_rotation_effect())
        self._register_effect("trail", "Trails", self._build_single_slider("trail_strength", 0.0, 0.9, 0.12))
        self._register_effect("flicker", "Flicker", self._build_band_effect("flicker_amount", 0.0, 1.0, 0.08, "all"))
        self._register_effect("hue", "Hue Shift", self._build_band_effect("hue_shift_amount", 0.0, 180.0, 6.0, "all"))
        self._register_effect("scanline", "Scanline", self._build_scanline_effect(default_amount=0.6, default_band="all"))
        self._register_effect("decimate", "Decimate", self._build_decimate_effect(default_step=1))
        self._register_effect("jitter", "Jitter", self._build_band_effect("jitter_amount", 0.0, 5.0, 0.25, "all"))
        self._register_effect("dither", "Dither", self._build_single_slider("dither_amount", 0.0, 1.0, 0.2))
        self._register_effect("phosphor", "Phosphor Mask", self._build_single_slider("phosphor_amount", 0.0, 1.0, 0.35))
        self._register_effect("bloom", "Bloom", self._build_bloom_effect())
        self._register_effect("vignette", "Vignette", self._build_vignette_effect())
        self._register_effect("chromatic", "Chromatic Aberration", self._build_chromatic_effect())
        self._register_effect("barrel", "Barrel Distortion", self._build_barrel_effect())
        self._register_effect("noise", "Noise", self._build_single_slider("noise_amount", 0.0, 1.0, 0.15))
        self._register_effect("h_jitter", "Horizontal Jitter", self._build_h_jitter_effect())
        self._register_effect("v_roll", "Vertical Roll", self._build_v_roll_effect())
        self._register_effect("bleed", "Color Bleed", self._build_single_slider("bleed_amount", 0.0, 1.0, 0.25))

        for key in self.default_effects:
            self._activate_effect(key)

    def _register_effect(self, key: str, title: str, content: QWidget) -> None:
        effect = EffectWidget(key, title, content)
        effect.removed.connect(self._remove_effect)
        self.effect_widgets[key] = effect
        self.effects_container.addWidget(effect)
        effect.setVisible(False)

    def _set_default(self, widget: QWidget, value: object) -> None:
        widget.setProperty("default_value", value)

    def _build_smoothing_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        self.smoothing_amount = QDoubleSpinBox()
        self.smoothing_amount.setRange(0.0, 1.0)
        self.smoothing_amount.setSingleStep(0.05)
        self.smoothing_amount.setValue(0.2)
        self._set_default(self.smoothing_amount, 0.2)
        form.addRow("Amount", self.smoothing_amount)
        self.smoothing_amount.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_displace_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        row_x = self._amount_band_row("mod_displace_x_amount", 0.0, 20.0, 6.0, "band:1")
        row_y = self._amount_band_row("mod_displace_y_amount", 0.0, 20.0, 6.0, "band:2")
        form.addRow("X", row_x)
        form.addRow("Y", row_y)
        return widget

    def _build_glow_combo(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        self.enable_glow = QCheckBox("Enable Glow")
        self.enable_glow.setChecked(True)
        self._set_default(self.enable_glow, True)
        self.enable_glow.toggled.connect(self._update_preview_frame)
        form.addRow(self.enable_glow)
        row_mod = self._amount_band_row("mod_glow_amount", 0.0, 5.0, 1.0, "all")
        form.addRow("Mod", row_mod)
        glow_radius = self._single_spin("glow_radius", 0.5, 8.0, 2.6)
        form.addRow("Radius", glow_radius)
        return widget

    def _build_warp_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        row_amount = self._amount_band_row("mod_warp_amount", 0.0, 20.0, 8.0, "band:4")
        row_speed = self._amount_band_row("mod_warp_speed_amount", 0.0, 10.0, 2.0, "band:2")
        form.addRow("Amount", row_amount)
        form.addRow("Speed", row_speed)
        return widget

    def _build_rotation_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        row = self._amount_band_row("mod_rotation_amount", 0.0, 360.0, 0.0, "all")
        form.addRow("Amount", row)
        direction = QComboBox()
        direction.addItem("Clockwise", "cw")
        direction.addItem("Counterclockwise", "ccw")
        direction.addItem("Alternate", "alternate")
        form.addRow("Direction", direction)
        direction.currentIndexChanged.connect(self._update_preview_frame)
        controls = self.effect_controls.setdefault("mod_rotation_amount", {})
        controls["direction"] = direction
        return widget

    def _build_band_effect(
        self,
        amount_key: str,
        min_value: float,
        max_value: float,
        default_amount: float,
        default_band: str,
    ) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        row = self._amount_band_row(amount_key, min_value, max_value, default_amount, default_band)
        form.addRow("Amount", row)
        return widget

    def _amount_band_row(
        self,
        amount_key: str,
        min_value: float,
        max_value: float,
        default_amount: float,
        default_band: str,
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        amount = QDoubleSpinBox()
        amount.setRange(min_value, max_value)
        amount.setSingleStep(0.1 if max_value > 1.0 else 0.01)
        amount.setValue(default_amount)
        self._set_default(amount, default_amount)
        band = self._band_combo()
        self._set_combo_by_data(band, default_band)
        self._set_default(band, default_band)
        layout.addWidget(amount)
        layout.addWidget(band, 1)
        self.effect_controls[amount_key] = {"amount": amount, "band": band}
        amount.valueChanged.connect(self._update_preview_frame)
        band.currentIndexChanged.connect(self._update_preview_frame)
        return row

    def _single_spin(self, key: str, low: float, high: float, value: float) -> QWidget:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        self._set_default(spin, value)
        self.effect_controls[key] = {"amount": spin}
        spin.valueChanged.connect(self._update_preview_frame)
        return spin

    def _build_scanline_effect(self, *, default_amount: float, default_band: str) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        row = self._amount_band_row("scanline_amount", 0.0, 10.0, default_amount, default_band)
        speed = QDoubleSpinBox()
        speed.setRange(0.1, 10.0)
        speed.setSingleStep(0.1)
        speed.setValue(1.0)
        self._set_default(speed, 1.0)
        form.addRow("Amount", row)
        form.addRow("Speed", speed)
        self.effect_controls["scanline_amount"]["speed"] = speed
        speed.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_decimate_effect(self, *, default_step: int) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        decimate = QSpinBox()
        decimate.setRange(1, 16)
        decimate.setValue(default_step)
        self._set_default(decimate, default_step)
        form.addRow("Step", decimate)
        self.effect_controls["decimate_step"] = {"step": decimate}
        decimate.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_single_slider(self, key: str, low: float, high: float, value: float) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        form.addRow("Amount", spin)
        self.effect_controls[key] = {"amount": spin}
        spin.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_bloom_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 1.0)
        amount.setSingleStep(0.05)
        amount.setValue(0.35)
        self._set_default(amount, 0.35)
        radius = QDoubleSpinBox()
        radius.setRange(0.5, 10.0)
        radius.setSingleStep(0.5)
        radius.setValue(2.5)
        self._set_default(radius, 2.5)
        threshold = QDoubleSpinBox()
        threshold.setRange(0.0, 1.0)
        threshold.setSingleStep(0.05)
        threshold.setValue(0.6)
        self._set_default(threshold, 0.6)
        form.addRow("Amount", amount)
        form.addRow("Radius", radius)
        form.addRow("Threshold", threshold)
        self.effect_controls["bloom_amount"] = {"amount": amount}
        self.effect_controls["bloom_radius"] = {"amount": radius}
        self.effect_controls["bloom_threshold"] = {"amount": threshold}
        amount.valueChanged.connect(self._update_preview_frame)
        radius.valueChanged.connect(self._update_preview_frame)
        threshold.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_vignette_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 1.0)
        amount.setSingleStep(0.05)
        amount.setValue(0.35)
        self._set_default(amount, 0.35)
        power = QDoubleSpinBox()
        power.setRange(0.5, 3.0)
        power.setSingleStep(0.1)
        power.setValue(1.8)
        self._set_default(power, 1.8)
        form.addRow("Amount", amount)
        form.addRow("Power", power)
        self.effect_controls["vignette_amount"] = {"amount": amount}
        self.effect_controls["vignette_power"] = {"amount": power}
        amount.valueChanged.connect(self._update_preview_frame)
        power.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_chromatic_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        shift_x = QDoubleSpinBox()
        shift_x.setRange(-5.0, 5.0)
        shift_x.setSingleStep(0.1)
        shift_x.setValue(1.0)
        self._set_default(shift_x, 1.0)
        shift_y = QDoubleSpinBox()
        shift_y.setRange(-5.0, 5.0)
        shift_y.setSingleStep(0.1)
        shift_y.setValue(0.5)
        self._set_default(shift_y, 0.5)
        form.addRow("Shift X", shift_x)
        form.addRow("Shift Y", shift_y)
        self.effect_controls["chroma_shift_x"] = {"amount": shift_x}
        self.effect_controls["chroma_shift_y"] = {"amount": shift_y}
        shift_x.valueChanged.connect(self._update_preview_frame)
        shift_y.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_barrel_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        amount = QDoubleSpinBox()
        amount.setRange(-0.5, 0.5)
        amount.setSingleStep(0.02)
        amount.setValue(0.12)
        self._set_default(amount, 0.12)
        form.addRow("Amount", amount)
        self.effect_controls["barrel_amount"] = {"amount": amount}
        amount.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_h_jitter_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 20.0)
        amount.setSingleStep(0.5)
        amount.setValue(2.0)
        self._set_default(amount, 2.0)
        speed = QDoubleSpinBox()
        speed.setRange(0.1, 10.0)
        speed.setSingleStep(0.1)
        speed.setValue(2.0)
        self._set_default(speed, 2.0)
        form.addRow("Amount (px)", amount)
        form.addRow("Speed", speed)
        self.effect_controls["h_jitter_amount"] = {"amount": amount}
        self.effect_controls["h_jitter_speed"] = {"amount": speed}
        amount.valueChanged.connect(self._update_preview_frame)
        speed.valueChanged.connect(self._update_preview_frame)
        return widget

    def _build_v_roll_effect(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 50.0)
        amount.setSingleStep(1.0)
        amount.setValue(8.0)
        self._set_default(amount, 8.0)
        speed = QDoubleSpinBox()
        speed.setRange(0.05, 5.0)
        speed.setSingleStep(0.05)
        speed.setValue(0.35)
        self._set_default(speed, 0.35)
        form.addRow("Amount (px)", amount)
        form.addRow("Speed", speed)
        self.effect_controls["v_roll_amount"] = {"amount": amount}
        self.effect_controls["v_roll_speed"] = {"amount": speed}
        amount.valueChanged.connect(self._update_preview_frame)
        speed.valueChanged.connect(self._update_preview_frame)
        return widget

    def _band_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItem("Subs", "band:0")
        combo.addItem("Lows", "band:1")
        combo.addItem("Low Mids", "band:2")
        combo.addItem("High Mids", "band:3")
        combo.addItem("Highs", "band:4")
        combo.addItem("All", "all")
        combo.addItem("Oscillator", "osc")
        return combo

    def _set_combo_by_data(self, combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _activate_effect(self, key: str) -> None:
        widget = self.effect_widgets.get(key)
        if not widget:
            return
        widget.setVisible(True)
        self.active_effects.add(key)
        if key == "smoothing":
            self.preview_smoothed = None
        self._refresh_effects_dropdown()

    def _remove_effect(self, key: str) -> None:
        widget = self.effect_widgets.get(key)
        if not widget:
            return
        widget.setVisible(False)
        self.active_effects.discard(key)
        self._reset_effect_values(key)
        self._refresh_effects_dropdown()
        self._update_preview_frame()

    def _reset_effect_values(self, key: str) -> None:
        if key == "smoothing":
            self.preview_smoothed = None
        if key == "glow":
            default = self.enable_glow.property("default_value")
            self.enable_glow.setChecked(bool(default) if default is not None else False)
        for control_key in self.effect_groups.get(key, []):
            controls = self.effect_controls.get(control_key)
            if not controls:
                continue
            for name, widget in controls.items():
                if isinstance(widget, QDoubleSpinBox):
                    default = widget.property("default_value")
                    widget.setValue(float(default) if default is not None else 0.0)
                if isinstance(widget, QSpinBox):
                    default = widget.property("default_value")
                    widget.setValue(int(default) if default is not None else 1)
                if isinstance(widget, QComboBox):
                    default = widget.property("default_value")
                    if default is not None:
                        self._set_combo_by_data(widget, str(default))
                    elif name == "direction":
                        widget.setCurrentIndex(0)

    def _refresh_effects_dropdown(self) -> None:
        self.add_effect_combo.blockSignals(True)
        self.add_effect_combo.clear()
        self.add_effect_combo.addItem("Select effect...", "")
        for key, widget in self.effect_widgets.items():
            if key in self.active_effects:
                continue
            label = widget.findChild(QLabel)
            name = label.text() if label else key
            self.add_effect_combo.addItem(name, key)
        self.add_effect_combo.blockSignals(False)

    def _add_effect_from_combo(self) -> None:
        key = self.add_effect_combo.currentData()
        if not key:
            return
        self._activate_effect(key)
        self.add_effect_combo.setCurrentIndex(0)
        self._update_preview_frame()

    def _presets_dir(self) -> str:
        return os.path.join(self._app_root, "presets")

    def _collect_effect_preset(self) -> dict[str, object]:
        controls_payload: dict[str, dict[str, object]] = {}
        for key, controls in self.effect_controls.items():
            payload: dict[str, object] = {}
            for name, widget in controls.items():
                if isinstance(widget, QDoubleSpinBox):
                    payload[name] = float(widget.value())
                elif isinstance(widget, QSpinBox):
                    payload[name] = int(widget.value())
                elif isinstance(widget, QComboBox):
                    data = widget.currentData()
                    payload[name] = data if data is not None else widget.currentText()
                elif isinstance(widget, QCheckBox):
                    payload[name] = bool(widget.isChecked())
            if payload:
                controls_payload[key] = payload

        preset = {
            "version": 1,
            "active_effects": sorted(self.active_effects),
            "controls": controls_payload,
            "smoothing_amount": float(self.smoothing_amount.value()),
            "glow_enabled": bool(self.enable_glow.isChecked()),
        }
        return preset

    def _apply_effect_preset(self, preset: dict[str, object]) -> None:
        active = preset.get("active_effects", [])
        if not isinstance(active, list):
            raise ValueError("Preset active_effects must be a list")
        controls = preset.get("controls", {})
        if not isinstance(controls, dict):
            raise ValueError("Preset controls must be an object")

        for key in list(self.active_effects):
            self._remove_effect(key)

        for key in active:
            if key in self.effect_widgets:
                self._activate_effect(key)

        for key, values in controls.items():
            if key not in self.effect_controls or not isinstance(values, dict):
                continue
            for name, value in values.items():
                widget = self.effect_controls[key].get(name)
                if widget is None:
                    continue
                widget.blockSignals(True)
                try:
                    if isinstance(widget, QDoubleSpinBox):
                        widget.setValue(float(value))
                    elif isinstance(widget, QSpinBox):
                        widget.setValue(int(value))
                    elif isinstance(widget, QComboBox):
                        if value is None:
                            pass
                        else:
                            self._set_combo_by_data(widget, str(value))
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(bool(value))
                finally:
                    widget.blockSignals(False)

        smoothing_amount = preset.get("smoothing_amount")
        if isinstance(smoothing_amount, (int, float)):
            self.smoothing_amount.blockSignals(True)
            self.smoothing_amount.setValue(float(smoothing_amount))
            self.smoothing_amount.blockSignals(False)

        glow_enabled = preset.get("glow_enabled")
        if isinstance(glow_enabled, bool):
            self.enable_glow.blockSignals(True)
            self.enable_glow.setChecked(glow_enabled)
            self.enable_glow.blockSignals(False)

        self._refresh_effects_dropdown()
        self._update_preview_frame()

    def _save_preset(self) -> None:
        presets_dir = self._presets_dir()
        os.makedirs(presets_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preset",
            os.path.join(presets_dir, "preset.json"),
            "Preset Files (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = f"{path}.json"
        preset = self._collect_effect_preset()
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(preset, handle, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Preset Save Failed", str(exc))

    def _load_preset(self) -> None:
        presets_dir = self._presets_dir()
        os.makedirs(presets_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            presets_dir,
            "Preset Files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                preset = json.load(handle)
            if not isinstance(preset, dict):
                raise ValueError("Preset file must be a JSON object")
            self._apply_effect_preset(preset)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Preset Load Failed", str(exc))

    def _update_visibility(self) -> None:
        media_mode = self.media_mode_combo.currentData()
        audio_mode = self.audio_mode_combo.currentData()

        is_media = media_mode == "media"
        is_shapes = media_mode == "shapes"
        is_script = media_mode == "script"
        is_text = media_mode == "text"
        self.media_path.setEnabled(is_media)
        self.media_browse.setEnabled(is_media)
        self.shape_frame.setVisible(is_shapes)
        self.script_frame.setVisible(is_script)
        self.text_frame.setVisible(is_text)

        is_audio_file = audio_mode == "file"
        self.audio_path.setEnabled(is_audio_file)
        self.audio_browse.setEnabled(is_audio_file)
        osc_active = audio_mode == "osc"
        self.osc_frame.setVisible(osc_active)
        if not osc_active:
            self.osc_audio_monitor.setChecked(False)
            self._stop_osc_audio()
        self.osc_audio_monitor.setEnabled(osc_active)
        if osc_active:
            self._sync_duration_for_osc()

        self._update_shape_params()
        self._update_preview_frame()

    def _update_shape_params(self) -> None:
        shape = self.shape_combo.currentData()
        for key, widget in self.shape_param_widgets.items():
            widget.setVisible(key == shape)

    def _browse_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select media", "", "Media Files (*.*)")
        if path:
            self.media_path.setText(path)

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select audio", "", "Audio Files (*.*)")
        if path:
            self.audio_path.setText(path)
            self._load_audio_analysis(path)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Output MP4",
            self.output_path.text() or os.path.join("output", "output.mp4"),
            "MP4 Files (*.mp4)",
        )
        if path:
            self.output_path.setText(path)

    def _browse_script(self) -> None:
        root = os.path.join(os.getcwd(), "scripts")
        path, _ = QFileDialog.getOpenFileName(self, "Select script", root, "Python Files (*.py)")
        if path:
            self.script_path.setText(path)

    def _on_text_font_changed(self, family: str) -> None:
        if family:
            self.text_font_family = family
            self._update_preview_frame()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self._color_rgb()), self)
        if color.isValid():
            self._set_color(color)

    def _color_rgb(self) -> tuple[int, int, int]:
        return (self.r_spin.value(), self.g_spin.value(), self.b_spin.value())

    def _set_color(self, color: QColor) -> None:
        self.r_spin.blockSignals(True)
        self.g_spin.blockSignals(True)
        self.b_spin.blockSignals(True)
        self.r_spin.setValue(color.red())
        self.g_spin.setValue(color.green())
        self.b_spin.setValue(color.blue())
        self.r_spin.blockSignals(False)
        self.g_spin.blockSignals(False)
        self.b_spin.blockSignals(False)
        self.color_button.setStyleSheet(
            f"border-color: {ACCENT}; background-color: {color.name()};"
        )
        self._update_preview_frame()

    def _on_color_spin(self) -> None:
        color = QColor(self.r_spin.value(), self.g_spin.value(), self.b_spin.value())
        self.color_button.setStyleSheet(
            f"border-color: {ACCENT}; background-color: {color.name()};"
        )
        self._update_preview_frame()

    def _load_audio_analysis(self, path: str) -> None:
        if self.audio_worker and self.audio_worker.isRunning():
            self.audio_worker.quit()
            self.audio_worker.wait(100)
        self.audio_analysis = None
        self.audio_worker = AudioAnalysisWorker(path, self.preview_fps, 5)
        self.audio_worker.finished.connect(self._on_audio_analysis_ready)
        self.audio_worker.failed.connect(self._on_audio_analysis_failed)
        self.audio_worker.start()

    def _on_audio_analysis_ready(self, analysis: AudioAnalysis) -> None:
        self.audio_analysis = analysis
        self._update_preview_frame()

    def _on_audio_analysis_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Audio Error", message)

    def _reload_script(self) -> None:
        path = self.script_path.text().strip()
        if not path:
            self.script_generate = None
            return
        try:
            from .render import _load_script

            self.script_generate = _load_script(path)
        except Exception as exc:  # noqa: BLE001
            self.script_generate = None
            QMessageBox.warning(self, "Script Error", str(exc))
        self._update_preview_frame()

    def _toggle_play(self) -> None:
        if self.play_button.isChecked():
            self._start_playback()
        else:
            self._pause_playback()

    def _start_playback(self) -> None:
        self.play_button.setChecked(True)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        if self.audio_mode_combo.currentData() == "file":
            audio_path = self.audio_path.text().strip()
            if audio_path:
                self.media_player.setSource(QUrl.fromLocalFile(audio_path))
                self.media_player.play()
        self.preview_timer.start()

    def _pause_playback(self) -> None:
        self.play_button.setChecked(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        if self.audio_mode_combo.currentData() == "file":
            self.media_player.pause()
        self.preview_timer.stop()

    def _stop_playback(self) -> None:
        self._pause_playback()
        self.preview_time = self.loop_in_ms / 1000.0
        self.media_player.stop()
        self._sync_slider_from_time()
        self._update_preview_frame()

    def _toggle_loop(self) -> None:
        enabled = self.loop_button.isChecked()
        self.timeline.set_loop_enabled(enabled)
        if enabled and self.loop_out_ms <= self.loop_in_ms:
            self.loop_out_ms = self.timeline.maximum()
            self.timeline.set_loop(self.loop_in_ms, self.loop_out_ms)

    def _set_loop_in(self) -> None:
        self.loop_in_ms = self.timeline.value()
        if self.loop_out_ms <= self.loop_in_ms:
            self.loop_out_ms = self.timeline.maximum()
        self.timeline.set_loop(self.loop_in_ms, self.loop_out_ms)
        self._update_preview_labels(self.loop_in_ms, self.loop_out_ms)

    def _set_loop_out(self) -> None:
        self.loop_out_ms = max(self.loop_in_ms + 1, self.timeline.value())
        self.timeline.set_loop(self.loop_in_ms, self.loop_out_ms)
        self._update_preview_labels(self.loop_in_ms, self.loop_out_ms)

    def _toggle_mute(self) -> None:
        self.audio_output.setVolume(0.0 if self.mute_button.isChecked() else 0.6)

    def _on_loop_changed(self, loop_in: int, loop_out: int) -> None:
        self.loop_in_ms = loop_in
        self.loop_out_ms = loop_out
        self._update_preview_labels(loop_in, loop_out)

    def _on_preview_seek_start(self) -> None:
        self._preview_dragging = True

    def _on_preview_seek_end(self) -> None:
        self._preview_dragging = False
        self._seek_from_slider()

    def _on_preview_seek(self, value: int) -> None:
        if self._preview_dragging:
            self._update_preview_time(value)
            self.preview_time = value / 1000.0
            self._update_preview_frame()

    def _seek_from_slider(self) -> None:
        if self.audio_mode_combo.currentData() == "file":
            self.media_player.setPosition(self.timeline.value())
        self.preview_time = self.timeline.value() / 1000.0
        self._update_preview_frame()

    def _sync_slider_from_media(self, position: int) -> None:
        if not self.timeline.isSliderDown():
            self.timeline.setValue(position)
        self.preview_time = position / 1000.0
        self._update_preview_time(position)

    def _sync_duration_from_media(self, duration: int) -> None:
        self.timeline.setRange(0, max(1, duration))
        self.loop_out_ms = duration
        self.timeline.set_loop(self.loop_in_ms, self.loop_out_ms)
        self._update_preview_labels(self.loop_in_ms, self.loop_out_ms)

    def _sync_duration_for_osc(self) -> None:
        if self.audio_mode_combo.currentData() != "osc":
            return
        duration = int(max(0.5, float(self.osc_duration.value())) * 1000)
        self.timeline.setRange(0, max(1, duration))
        self.loop_out_ms = duration
        self.timeline.set_loop(self.loop_in_ms, self.loop_out_ms)
        self._update_preview_labels(self.loop_in_ms, self.loop_out_ms)

    def _sync_slider_from_time(self) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(self.preview_time * 1000))
        self.timeline.blockSignals(False)
        self._update_preview_time(int(self.preview_time * 1000))

    def _update_preview_time(self, position_ms: int) -> None:
        total_ms = self.timeline.maximum()
        self.preview_time_label.setText(
            f"{self._format_time(position_ms)} / {self._format_time(total_ms)}"
        )

    def _update_preview_labels(self, in_ms: int, out_ms: int) -> None:
        self.preview_in_label.setText(f"In: {self._format_time(in_ms)}")
        self.preview_out_label.setText(f"Out: {self._format_time(out_ms)}")
        self._update_preview_time(self.timeline.value())

    def _format_time(self, ms: int) -> str:
        seconds = max(0, int(ms // 1000))
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}:{secs:02d}"

    def _tick_preview(self) -> None:
        if self.audio_mode_combo.currentData() == "osc":
            self.preview_time += 1.0 / float(self.preview_fps)
            duration = max(0.5, float(self.osc_duration.value()))
            if self.loop_button.isChecked():
                if self.preview_time * 1000 >= self.loop_out_ms:
                    self.preview_time = self.loop_in_ms / 1000.0
            else:
                if self.preview_time >= duration:
                    self.preview_time = 0.0
                    self._pause_playback()
            self._sync_slider_from_time()
        else:
            if self.loop_button.isChecked() and self.media_player.position() >= self.loop_out_ms:
                self.media_player.setPosition(self.loop_in_ms)
        self._update_preview_frame()

    def _update_preview_buffer(self) -> None:
        size = self.preview_canvas.size()
        self.preview_width = max(320, size.width())
        self.preview_height = max(180, size.height())
        self._update_preview_frame()

    def _update_preview_frame(self) -> None:
        settings = self._collect_settings()
        width = self.preview_width
        height = self.preview_height

        if settings.audio_mode == "file" and self.audio_analysis is None:
            audio_path = settings.audio_path
            if audio_path:
                self._load_audio_analysis(audio_path)

        bands = np.zeros(5, dtype=np.float32)
        if settings.audio_mode == "osc":
            osc = _oscillator_value(settings, self.preview_time)
            bands[:] = osc
        else:
            if self.audio_analysis:
                frame_index = int(self.preview_time * self.preview_fps)
                bands = band_at_frame(self.audio_analysis.band_energies, frame_index)
            osc = _oscillator_value(settings, self.preview_time)

        if settings.smoothing_enabled:
            if self.preview_smoothed is None:
                self.preview_smoothed = bands.astype(np.float32)
            else:
                alpha = max(0.0, min(1.0, settings.smoothing_amount))
                self.preview_smoothed = self.preview_smoothed + (bands - self.preview_smoothed) * alpha
            bands = self.preview_smoothed

        osc = _oscillator_value(settings, self.preview_time)

        sig_displace_x = _mod_value(bands, settings.mod_displace_x_band, osc, settings.osc_mix)
        sig_displace_y = _mod_value(bands, settings.mod_displace_y_band, osc, settings.osc_mix)
        sig_thickness = _mod_value(bands, settings.mod_thickness_band, osc, settings.osc_mix)
        sig_glow = _mod_value(bands, settings.mod_glow_band, osc, settings.osc_mix)
        sig_threshold = _mod_value(bands, settings.mod_threshold_band, osc, settings.osc_mix)
        sig_warp = _mod_value(bands, settings.mod_warp_band, osc, settings.osc_mix)
        sig_warp_speed = _mod_value(bands, settings.mod_warp_speed_band, osc, settings.osc_mix)
        sig_rotation = _mod_value(bands, settings.mod_rotation_band, osc, settings.osc_mix)
        sig_rotation *= _rotation_direction(settings.mod_rotation_direction, self.preview_time)
        sig_flicker = _mod_value(bands, settings.flicker_band, osc, settings.osc_mix)
        sig_hue = _mod_value(bands, settings.hue_shift_band, osc, settings.osc_mix)
        sig_scan = _mod_value(bands, settings.scanline_band, osc, settings.osc_mix)
        sig_jitter = _mod_value(bands, settings.jitter_band, osc, settings.osc_mix)

        energy = float(bands.mean()) if bands.size else 0.0
        hue_shift = settings.hue_shift_amount * sig_hue
        shifted_color = _apply_hue_shift(settings.color_rgb, hue_shift)
        color = QColor(*shifted_color)

        if settings.media_mode == "text":
            try:
                polylines = _text_to_polylines(
                    self.text_input.text(),
                    font_family=self.text_font_family,
                    scale=float(self.text_scale.value()),
                )
            except RuntimeError as exc:
                polylines = None
                if getattr(self, "_text_error", None) != str(exc):
                    self._text_error = str(exc)
                    QMessageBox.warning(self, "Text Error", str(exc))
        elif settings.media_mode == "script" and self.script_generate:
            payload = _script_audio_payload(bands, osc)
            polylines = self.script_generate(
                self.preview_time,
                payload,
                {"width": width, "height": height, "fps": self.preview_fps},
            )
        else:
            polylines = None

        shape_type = settings.shape_type if settings.media_mode == "shapes" else "ring"
        base_rotation = float(settings.shape_rotation)
        if settings.mod_rotation_amount != 0.0:
            base_rotation += settings.mod_rotation_amount * sig_rotation

        threshold_value = (
            max(0.0, settings.edge_threshold - settings.mod_threshold_amount * sig_threshold)
            if "threshold" in self.active_effects
            else 0.0
        )
        shape_params = {
            "star_points": settings.star_points,
            "star_inner": settings.star_inner,
            "rect_width": settings.rect_width,
            "rect_height": settings.rect_height,
            "ellipse_x": settings.ellipse_x,
            "ellipse_y": settings.ellipse_y,
            "spiral_turns": settings.spiral_turns,
            "spiral_growth": settings.spiral_growth,
            "lemniscate_scale": settings.lemniscate_scale,
            "cardioid_scale": settings.cardioid_scale,
            "clover_petals": settings.clover_petals,
            "clover_scale": settings.clover_scale,
            "superellipse_n": settings.superellipse_n,
            "superellipse_scale": settings.superellipse_scale,
            "glow_radius": settings.glow_radius,
            "trail_strength": settings.trail_strength,
            "flicker_amount": settings.flicker_amount * sig_flicker,
            "hue_shift_amount": settings.hue_shift_amount * sig_hue,
            "scanline_amount": settings.scanline_amount * sig_scan,
            "scanline_speed": settings.scanline_speed,
            "decimate_step": settings.decimate_step,
            "jitter_amount": settings.jitter_amount * sig_jitter,
            "threshold": threshold_value,
            "time": self.preview_time,
            "poly_rotation": base_rotation,
            "dither_amount": settings.dither_amount,
            "phosphor_amount": settings.phosphor_amount,
            "bloom_amount": settings.bloom_amount,
            "bloom_radius": settings.bloom_radius,
            "bloom_threshold": settings.bloom_threshold,
            "vignette_amount": settings.vignette_amount,
            "vignette_power": settings.vignette_power,
            "chroma_shift_x": settings.chroma_shift_x,
            "chroma_shift_y": settings.chroma_shift_y,
            "barrel_amount": settings.barrel_amount,
            "noise_amount": settings.noise_amount,
            "h_jitter_amount": settings.h_jitter_amount,
            "h_jitter_speed": settings.h_jitter_speed,
            "v_roll_amount": settings.v_roll_amount,
            "v_roll_speed": settings.v_roll_speed,
            "bleed_amount": settings.bleed_amount,
        }

        displace_x = settings.mod_displace_x_amount * sig_displace_x
        displace_y = settings.mod_displace_y_amount * sig_displace_y
        thickness = max(2.0, 1.0 + settings.mod_thickness_amount * sig_thickness)
        warp_amount = settings.mod_warp_amount * sig_warp
        warp_speed = 1.0 + settings.mod_warp_speed_amount * sig_warp_speed
        warp_phase = self.preview_time * warp_speed * 6.0
        glow_strength = max(0.0, settings.glow_strength + settings.mod_glow_amount * sig_glow)

        self.preview_canvas.update_state(
            color=color,
            energy=energy,
            displace_x=displace_x * 6.0,
            displace_y=displace_y * 6.0,
            thickness=thickness,
            warp_amount=warp_amount,
            warp_phase=warp_phase,
            shape_type=shape_type,
            polygon_sides=settings.polygon_sides,
            shape_rotation=base_rotation,
            shape_size=settings.shape_size,
            shape_params=shape_params,
            polylines=polylines,
            preserve_aspect=settings.preserve_aspect,
            glow_strength=glow_strength,
        )

    def _collect_settings(self) -> RenderSettings:
        media_mode = self.media_mode_combo.currentData()
        audio_mode = self.audio_mode_combo.currentData()
        shape_type = self.shape_combo.currentData()
        color = self._color_rgb()

        settings = RenderSettings(
            media_path=self.media_path.text().strip(),
            audio_path=self.audio_path.text().strip(),
            output_path=self.output_path.text().strip(),
            audio_mode=audio_mode,
            osc_duration=float(self.osc_duration.value()),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            fps=self.fps_spin.value(),
            enable_line=self.enable_line.isChecked(),
            enable_lissajous=self.enable_lissajous.isChecked(),
            enable_glow=self.enable_glow.isChecked() if "glow" in self.active_effects else False,
            bands=5,
            preserve_aspect=self.preserve_aspect.isChecked(),
            edge_mode=self.render_mode_combo.currentData(),
            edge_method=self.edge_method_combo.currentData(),
            edge_threshold=float(self.edge_threshold.value()),
            glow_strength=float(self.glow_strength.value()),
            color_rgb=color,
            smoothing_enabled="smoothing" in self.active_effects,
            smoothing_amount=float(self.smoothing_amount.value()) if "smoothing" in self.active_effects else 0.0,
            mod_displace_x_amount=self._effect_amount("mod_displace_x_amount", "displace"),
            mod_displace_x_band=self._effect_band("mod_displace_x_amount", "displace"),
            mod_displace_y_amount=self._effect_amount("mod_displace_y_amount", "displace"),
            mod_displace_y_band=self._effect_band("mod_displace_y_amount", "displace"),
            mod_thickness_amount=self._effect_amount("mod_thickness_amount", "thickness"),
            mod_thickness_band=self._effect_band("mod_thickness_amount", "thickness"),
            mod_glow_amount=self._effect_amount("mod_glow_amount", "glow"),
            mod_glow_band=self._effect_band("mod_glow_amount", "glow"),
            mod_threshold_amount=self._effect_amount("mod_threshold_amount", "threshold"),
            mod_threshold_band=self._effect_band("mod_threshold_amount", "threshold"),
            mod_warp_amount=self._effect_amount("mod_warp_amount", "warp"),
            mod_warp_band=self._effect_band("mod_warp_amount", "warp"),
            mod_warp_speed_amount=self._effect_amount("mod_warp_speed_amount", "warp"),
            mod_warp_speed_band=self._effect_band("mod_warp_speed_amount", "warp"),
            media_mode=media_mode,
            shape_type=shape_type,
            polygon_sides=self._shape_param_value("polygon", 0, default=5),
            shape_rotation=float(self.shape_rotation.value()),
            shape_size=float(self.shape_size.value()),
            star_points=self._shape_param_value("star", 0, default=5),
            star_inner=self._shape_param_value("star", 1, default=0.45),
            rect_width=self._shape_param_value("rectangle", 0, default=1.0),
            rect_height=self._shape_param_value("rectangle", 1, default=0.6),
            ellipse_x=self._shape_param_value("ellipse", 0, default=1.2),
            ellipse_y=self._shape_param_value("ellipse", 1, default=0.8),
            spiral_turns=self._shape_param_value("spiral", 0, default=3.5),
            spiral_growth=self._shape_param_value("spiral", 1, default=1.0),
            lemniscate_scale=self._shape_param_value("lemniscate", 0, default=1.4),
            cardioid_scale=self._shape_param_value("cardioid", 0, default=1.0),
            clover_petals=self._shape_param_value("clover", 0, default=4),
            clover_scale=self._shape_param_value("clover", 1, default=1.2),
            superellipse_n=self._shape_param_value("superellipse", 0, default=2.6),
            superellipse_scale=self._shape_param_value("superellipse", 1, default=1.1),
            mod_rotation_amount=self._effect_amount("mod_rotation_amount", "rotation"),
            mod_rotation_band=self._effect_band("mod_rotation_amount", "rotation"),
            mod_rotation_direction=self._effect_direction("mod_rotation_amount"),
            script_path=self.script_path.text().strip(),
            trail_strength=self._effect_amount("trail_strength", "trail"),
            glow_radius=self._effect_amount("glow_radius", "glow_radius"),
            flicker_amount=self._effect_amount("flicker_amount", "flicker"),
            flicker_band=self._effect_band("flicker_amount", "flicker"),
            hue_shift_amount=self._effect_amount("hue_shift_amount", "hue"),
            hue_shift_band=self._effect_band("hue_shift_amount", "hue"),
            scanline_amount=self._effect_amount("scanline_amount", "scanline"),
            scanline_band=self._effect_band("scanline_amount", "scanline"),
            scanline_speed=self._effect_speed("scanline_amount"),
            decimate_step=self._effect_step("decimate_step"),
            jitter_amount=self._effect_amount("jitter_amount", "jitter"),
            jitter_band=self._effect_band("jitter_amount", "jitter"),
            osc_waveform=self.osc_waveform.currentData(),
            osc_frequency=float(self.osc_frequency.value()),
            osc_depth=float(self.osc_depth.value()),
            osc_mix=float(self.osc_mix.value()),
            text_value=self.text_input.text(),
            text_scale=float(self.text_scale.value()),
            text_font_family=self.text_font_family,
            dither_amount=self._effect_amount("dither_amount", "dither"),
            phosphor_amount=self._effect_amount("phosphor_amount", "phosphor"),
            bloom_amount=self._effect_amount("bloom_amount", "bloom"),
            bloom_radius=self._effect_amount("bloom_radius", "bloom"),
            bloom_threshold=self._effect_amount("bloom_threshold", "bloom"),
            vignette_amount=self._effect_amount("vignette_amount", "vignette"),
            vignette_power=self._effect_amount("vignette_power", "vignette"),
            chroma_shift_x=self._effect_amount("chroma_shift_x", "chromatic"),
            chroma_shift_y=self._effect_amount("chroma_shift_y", "chromatic"),
            barrel_amount=self._effect_amount("barrel_amount", "barrel"),
            noise_amount=self._effect_amount("noise_amount", "noise"),
            h_jitter_amount=self._effect_amount("h_jitter_amount", "h_jitter"),
            h_jitter_speed=self._effect_amount("h_jitter_speed", "h_jitter"),
            v_roll_amount=self._effect_amount("v_roll_amount", "v_roll"),
            v_roll_speed=self._effect_amount("v_roll_speed", "v_roll"),
            bleed_amount=self._effect_amount("bleed_amount", "bleed"),
        )
        return settings

    def _shape_param_value(self, shape: str, index: int, *, default: float) -> float:
        widget = self.shape_param_widgets.get(shape)
        if not widget:
            return default
        form = widget.layout()
        if not form or index >= form.rowCount():
            return default
        field = form.itemAt(index, QFormLayout.FieldRole).widget()
        if isinstance(field, (QSpinBox, QDoubleSpinBox)):
            return field.value()
        return default

    def _effect_amount(self, key: str, effect_key: str) -> float:
        if effect_key not in self.active_effects:
            return 0.0
        controls = self.effect_controls.get(key)
        if not controls:
            return 0.0
        widget = controls.get("amount")
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        return 0.0

    def _effect_band(self, key: str, effect_key: str) -> str:
        if effect_key not in self.active_effects:
            return "all"
        controls = self.effect_controls.get(key)
        if not controls:
            return "all"
        widget = controls.get("band")
        if isinstance(widget, QComboBox):
            return widget.currentData()
        return "all"

    def _effect_direction(self, key: str) -> str:
        controls = self.effect_controls.get(key)
        if not controls:
            return "cw"
        widget = controls.get("direction")
        if isinstance(widget, QComboBox):
            return widget.currentData()
        return "cw"

    def _effect_speed(self, key: str) -> float:
        controls = self.effect_controls.get(key)
        if not controls or "speed" not in controls:
            return 1.0
        widget = controls.get("speed")
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        return 1.0

    def _effect_step(self, key: str) -> int:
        controls = self.effect_controls.get(key)
        if not controls:
            return 1
        widget = controls.get("step")
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        return 1

    def _start_render(self) -> None:
        settings = self._collect_settings()
        if settings.media_mode == "media" and not settings.media_path:
            QMessageBox.warning(self, "Missing media", "Select a media file.")
            return
        if settings.audio_mode == "file" and not settings.audio_path:
            QMessageBox.warning(self, "Missing audio", "Select an audio file.")
            return
        if settings.media_mode == "script" and not settings.script_path:
            QMessageBox.warning(self, "Missing script", "Select a script file.")
            return

        self.render_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.progress.setValue(0)

        self.render_worker = RenderWorker(settings)
        self.render_worker.progress.connect(self._render_progress)
        self.render_worker.finished.connect(self._render_finished)
        self.render_worker.canceled.connect(self._render_canceled)
        self.render_worker.failed.connect(self._render_failed)
        self.render_worker.start()

    def _cancel_render(self) -> None:
        if hasattr(self, "render_worker"):
            self.render_worker.cancel()

    def _render_progress(self, value: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(int((value / total) * 100))
        else:
            self.progress.setValue(value)

    def _render_finished(self) -> None:
        self.render_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.progress.setValue(100)
        QMessageBox.information(self, "Render complete", "Render finished successfully.")

    def _render_canceled(self) -> None:
        self.render_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.progress.setValue(0)

    def _render_failed(self, message: str) -> None:
        self.render_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.progress.setValue(0)
        QMessageBox.warning(self, "Render failed", message)

    def _start_osc_audio(self) -> None:
        if self.osc_audio_sink:
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(44100)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)
        self.osc_audio_device = OscillatorAudioDevice(
            lambda: self.osc_waveform.currentData(),
            lambda: float(self.osc_frequency.value()),
        )
        self.osc_audio_sink = QAudioSink(fmt, self)
        self.osc_audio_sink.setVolume(0.08)
        self.osc_audio_device.start()
        self.osc_audio_sink.start(self.osc_audio_device)

    def _stop_osc_audio(self) -> None:
        if self.osc_audio_sink:
            self.osc_audio_sink.stop()
            self.osc_audio_sink = None
        if self.osc_audio_device:
            self.osc_audio_device.stop()
            self.osc_audio_device = None

    def _toggle_osc_audio(self) -> None:
        if self.osc_audio_monitor.isChecked():
            self._start_osc_audio()
        else:
            self._stop_osc_audio()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key_Escape:
            if QMessageBox.question(self, "Exit", "Are you sure you want to exit?") == QMessageBox.Yes:
                QApplication.instance().quit()
            return
        super().keyPressEvent(event)
