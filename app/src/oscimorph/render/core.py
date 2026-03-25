from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageSequenceClip

from ..audio import AudioAnalysis, band_at_frame, frame_count, load_and_analyze
from ..runtime import temp_dir
from .modulation import (
    _apply_hue_shift,
    _mod_value,
    _oscillator_value,
    _rotation_direction,
    _script_audio_payload,
)
from .postfx import (
    _apply_barrel_distortion,
    _apply_bloom,
    _apply_chromatic_aberration,
    _apply_color_bleed,
    _apply_color_bleed_advanced,
    _apply_dither,
    _flicker_factor,
    _apply_scanlines,
    _apply_trail,
    _apply_horizontal_jitter,
    _apply_noise,
    _apply_phosphor_mask,
    _apply_vertical_roll,
    _apply_vignette,
)
from .progress import MoviepyLogger, ProgressTracker, ensure_temp_dir, init_progress_log
from .settings import RenderCancelled, RenderSettings
from .text import _load_script, _make_script_frame, _rotate_polylines, _text_to_polylines, _warp_polylines


def _resize_frame(
    frame: Image.Image,
    *,
    width: int,
    height: int,
    preserve_aspect: bool,
) -> Image.Image:
    if not preserve_aspect:
        return frame.resize((width, height), Image.LANCZOS)

    src_w, src_h = frame.size
    if src_w == 0 or src_h == 0:
        return Image.new("RGB", (width, height), (0, 0, 0))

    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = frame.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _load_media_frames(
    path: str,
    width: int,
    height: int,
    *,
    preserve_aspect: bool,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    ext = os.path.splitext(path)[1].lower()
    image_exts = {".gif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    if ext in video_exts:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open video")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frame = _resize_frame(
                    pil_frame,
                    width=width,
                    height=height,
                    preserve_aspect=preserve_aspect,
                )
                frames.append(np.array(pil_frame))
        finally:
            cap.release()
        return frames

    if ext in image_exts:
        with Image.open(path) as img:
            index = 0
            while True:
                img.seek(index)
                frame = img.convert("RGB")
                frame = _resize_frame(
                    frame,
                    width=width,
                    height=height,
                    preserve_aspect=preserve_aspect,
                )
                frames.append(np.array(frame))
                index += 1
                if not getattr(img, "is_animated", False) or index >= getattr(img, "n_frames", 1):
                    break
        return frames

    raise RuntimeError("Unsupported media format")


def _transform_shape_points(
    points: np.ndarray,
    *,
    decimate_step: int,
    jitter_amount: float,
    jitter_axis: str,
    jitter_style: str,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    step = max(1, int(decimate_step))
    if step > 1:
        pts = pts[::step]
    if pts.size == 0:
        return pts.astype(np.int32)
    if jitter_amount > 0.0:
        jitter = (np.random.rand(pts.shape[0], 2) - 0.5) * 2.0 * jitter_amount
        if jitter_style == "stepped":
            jitter = np.round(jitter)
        if jitter_axis == "x":
            jitter[:, 1] = 0.0
        elif jitter_axis == "y":
            jitter[:, 0] = 0.0
        pts = pts + jitter
    return pts.astype(np.int32)


def _make_shape_frame(
    *,
    width: int,
    height: int,
    shape_type: str,
    polygon_sides: int,
    shape_rotation: float,
    shape_size: float,
    star_points: int,
    star_inner: float,
    rect_width: float,
    rect_height: float,
    ellipse_x: float,
    ellipse_y: float,
    spiral_turns: float,
    spiral_growth: float,
    lemniscate_scale: float,
    cardioid_scale: float,
    clover_petals: int,
    clover_scale: float,
    superellipse_n: float,
    superellipse_scale: float,
    decimate_step: int,
    jitter_amount: float,
    jitter_axis: str,
    jitter_style: str,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.28 * float(shape_size))
    color = (255, 255, 255)
    rotation = float(shape_rotation) * np.pi / 180.0

    if shape_type == "polygon":
        sides = max(3, int(polygon_sides))
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        angles = angles + rotation
        points = np.stack(
            [
                center[0] + np.cos(angles) * radius,
                center[1] + np.sin(angles) * radius,
            ],
            axis=1,
        )
        points = _transform_shape_points(
            points,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "ellipse":
        angles = np.linspace(0, 2 * np.pi, 361)
        pts = np.stack(
            [
                np.cos(angles) * radius * ellipse_x,
                np.sin(angles) * radius * ellipse_y,
            ],
            axis=1,
        )
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = pts @ rot.T
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "rectangle":
        corners = np.array(
            [
                (-rect_width, -rect_height),
                (rect_width, -rect_height),
                (rect_width, rect_height),
                (-rect_width, rect_height),
                (-rect_width, -rect_height),
            ],
            dtype=np.float32,
        )
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = (corners @ rot.T) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "star":
        spikes = max(3, int(star_points))
        points = []
        for i in range(spikes * 2 + 1):
            t = (i / (spikes * 2)) * 2 * np.pi + rotation
            r = 1.0 if i % 2 == 0 else float(star_inner)
            points.append([np.cos(t) * r, np.sin(t) * r])
        pts = np.array(points, dtype=np.float32) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "spiral":
        turns = float(spiral_turns)
        steps = 360
        points = []
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi * turns
            r = (i / steps) * float(spiral_growth)
            points.append([np.cos(t) * r, np.sin(t) * r])
        pts = np.array(points, dtype=np.float32)
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = (pts @ rot.T) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "lemniscate":
        steps = 360
        points = []
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi
            denom = 1 + np.sin(t) ** 2
            x = np.cos(t) / denom
            y = (np.sin(t) * np.cos(t)) / denom
            points.append([x * float(lemniscate_scale), y * float(lemniscate_scale)])
        pts = np.array(points, dtype=np.float32)
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = (pts @ rot.T) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "cardioid":
        steps = 360
        points = []
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi
            r = (1 - np.cos(t)) * float(cardioid_scale)
            points.append([r * np.cos(t), r * np.sin(t)])
        pts = np.array(points, dtype=np.float32)
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = (pts @ rot.T) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "clover":
        steps = 360
        points = []
        petals = max(3, int(clover_petals))
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi
            r = np.cos(petals * t)
            points.append([r * np.cos(t) * float(clover_scale), r * np.sin(t) * float(clover_scale)])
        pts = np.array(points, dtype=np.float32)
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = (pts @ rot.T) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "superellipse":
        steps = 360
        n = max(0.5, float(superellipse_n))
        points = []
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi
            x = np.copysign(abs(np.cos(t)) ** (2 / n), np.cos(t)) * float(superellipse_scale)
            y = np.copysign(abs(np.sin(t)) ** (2 / n), np.sin(t)) * float(superellipse_scale)
            points.append([x, y])
        pts = np.array(points, dtype=np.float32)
        rot = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
            dtype=np.float32,
        )
        pts = (pts @ rot.T) * radius
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "heart":
        points = []
        steps = 360
        for i in range(steps + 1):
            t = (i / steps) * 2 * np.pi
            x = 16 * np.sin(t) ** 3
            y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
            x *= 0.03
            y *= 0.03
            xr = x * np.cos(rotation) - y * np.sin(rotation)
            yr = x * np.sin(rotation) + y * np.cos(rotation)
            points.append([center[0] + xr * radius * 2.0, center[1] - yr * radius * 2.0])
        pts = np.array(points, dtype=np.int32)
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    else:
        angles = np.linspace(0, 2 * np.pi, 361)
        pts = np.stack(
            [
                center[0] + np.cos(angles + rotation) * radius,
                center[1] + np.sin(angles + rotation) * radius,
            ],
            axis=1,
        )
        pts = _transform_shape_points(
            pts,
            decimate_step=decimate_step,
            jitter_amount=jitter_amount,
            jitter_axis=jitter_axis,
            jitter_style=jitter_style,
        )
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

    return canvas


def _audio_segment(audio: np.ndarray, center: int, size: int) -> np.ndarray:
    start = max(0, center - size // 2)
    end = min(audio.shape[1], center + size // 2)
    segment = audio[:, start:end]
    if segment.shape[1] < size:
        pad = size - segment.shape[1]
        segment = np.pad(segment, ((0, 0), (0, pad)), mode="constant")
    return segment


def _make_waveform(segment: np.ndarray, width: int) -> np.ndarray:
    mono = segment.mean(axis=0)
    if mono.size == 0:
        return np.zeros(width)
    mono = mono / (np.max(np.abs(mono)) + 1e-6)
    idx = np.linspace(0, mono.size - 1, width).astype(np.int32)
    return mono[idx]


def _make_lissajous(segment: np.ndarray, points: int) -> np.ndarray:
    left = segment[0]
    right = segment[1]
    if left.size == 0:
        return np.zeros((points, 2))
    left = left / (np.max(np.abs(left)) + 1e-6)
    right = right / (np.max(np.abs(right)) + 1e-6)
    idx = np.linspace(0, left.size - 1, points).astype(np.int32)
    return np.stack([left[idx], right[idx]], axis=1)


def _color_with_energy(color_rgb: tuple[int, int, int], energy: float) -> tuple[int, int, int]:
    energy = float(max(0.0, min(1.0, energy)))
    boost = 0.4 + 0.6 * energy
    r = int(min(255, color_rgb[0] * boost))
    g = int(min(255, color_rgb[1] * boost))
    b = int(min(255, color_rgb[2] * boost))
    return (b, g, r)


def _transform_edge_mask(
    mask: np.ndarray,
    *,
    decimate_step: int,
    jitter_amount: float,
    jitter_axis: str,
    jitter_style: str,
) -> np.ndarray:
    step = max(1, int(decimate_step))
    jitter_amount = max(0.0, float(jitter_amount))
    if step <= 1 and jitter_amount <= 0.0:
        return mask

    binary = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask

    out = np.zeros_like(binary)
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] < 2:
            continue
        if step > 1:
            points = points[::step]
            if points.shape[0] < 2:
                continue
        if jitter_amount > 0.0:
            jitter = (np.random.rand(points.shape[0], 2) - 0.5) * 2.0 * jitter_amount
            if jitter_style == "stepped":
                jitter = np.round(jitter)
            if jitter_axis == "x":
                jitter[:, 1] = 0.0
            elif jitter_axis == "y":
                jitter[:, 0] = 0.0
            points = points + jitter
        pts = points.astype(np.int32)
        is_closed = np.linalg.norm(points[0] - points[-1]) <= 1.5
        cv2.polylines(out, [pts], isClosed=is_closed, color=255, thickness=1, lineType=cv2.LINE_AA)

    return out.astype(np.float32) / 255.0


def _edge_oscilloscope(
    frame_rgb: np.ndarray,
    color_rgb: tuple[int, int, int],
    *,
    method: str,
    threshold: float,
    glow_strength: float,
    thickness: int,
    glow_radius: float,
    glow_threshold: float,
    glow_blend: str,
    decimate_step: int = 1,
    jitter_amount: float = 0.0,
    jitter_axis: str = "xy",
    jitter_style: str = "random",
) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    if method == "canny":
        low = max(0, int(threshold * 255))
        high = min(255, low * 3)
        edges = cv2.Canny(gray, low, high).astype(np.float32) / 255.0
        mag = edges
    else:
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobel_x, sobel_y)
        max_val = float(mag.max())
        if max_val > 0:
            mag = mag / max_val
        mag = np.clip((mag - threshold) / max(1e-6, 1 - threshold), 0, 1)

    mag = _transform_edge_mask(
        mag,
        decimate_step=decimate_step,
        jitter_amount=jitter_amount,
        jitter_axis=jitter_axis,
        jitter_style=jitter_style,
    )

    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        mag = cv2.dilate(mag, kernel, iterations=1)

    core = mag
    glow = cv2.GaussianBlur(mag, (0, 0), sigmaX=max(0.5, glow_radius))
    glow = np.clip((glow - glow_threshold) / max(1e-6, 1.0 - glow_threshold), 0, 1)

    if glow_blend == "screen":
        intensity = 1.0 - (1.0 - core) * (1.0 - glow * glow_strength)
    elif glow_blend == "soft":
        intensity = np.clip(core * 0.85 + glow * glow_strength * 0.7, 0, 1)
    else:
        intensity = np.clip(core + glow * glow_strength, 0, 1)
    color = np.array(color_rgb, dtype=np.float32) / 255.0
    rgb = (intensity[..., None] * color * 255.0).astype(np.uint8)
    return rgb


def _draw_glow_line(img_bgr: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    overlay = img_bgr.copy()
    cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=6, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0, dst=img_bgr)


def _draw_waveform(
    img_bgr: np.ndarray,
    waveform: np.ndarray,
    band_values: np.ndarray,
    color_rgb: tuple[int, int, int],
    *,
    enable_glow: bool,
    decimate_step: int,
    jitter_amount: float,
    jitter_axis: str,
    jitter_style: str,
) -> None:
    height, width = img_bgr.shape[:2]
    energy = float(band_values.mean())
    amplitude = (0.25 + 0.75 * energy) * (height * 0.35)
    center_y = int(height * 0.5)

    xs = np.arange(width)
    ys = (center_y + waveform * amplitude).astype(np.float32)
    if decimate_step > 1:
        xs = xs[::decimate_step]
        ys = ys[::decimate_step]
    if jitter_amount > 0.0:
        jitter = (np.random.rand(ys.size) - 0.5) * 2.0 * jitter_amount
        if jitter_style == "stepped":
            jitter = np.round(jitter)
        if jitter_axis in {"xy", "y"}:
            ys = ys + jitter
        if jitter_axis in {"xy", "x"}:
            xs = xs + jitter.astype(np.int32)
    ys = ys.astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    color = _color_with_energy(color_rgb, energy)

    if enable_glow:
        _draw_glow_line(img_bgr, pts, color)
    cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def _draw_lissajous(
    img_bgr: np.ndarray,
    lissajous: np.ndarray,
    band_values: np.ndarray,
    color_rgb: tuple[int, int, int],
    *,
    enable_glow: bool,
    decimate_step: int,
    jitter_amount: float,
    jitter_axis: str,
    jitter_style: str,
) -> None:
    height, width = img_bgr.shape[:2]
    points = lissajous.copy()
    if decimate_step > 1:
        points = points[::decimate_step]
    points[:, 0] = (points[:, 0] * 0.5 + 0.5) * (width - 1)
    points[:, 1] = (points[:, 1] * 0.5 + 0.5) * (height - 1)
    if jitter_amount > 0.0:
        jitter = (np.random.rand(points.shape[0], 2) - 0.5) * 2.0 * jitter_amount
        if jitter_style == "stepped":
            jitter = np.round(jitter)
        if jitter_axis == "x":
            jitter[:, 1] = 0.0
        elif jitter_axis == "y":
            jitter[:, 0] = 0.0
        points = points + jitter
    pts = points.astype(np.int32)
    energy = float(band_values.mean())
    color = _color_with_energy(color_rgb, energy)

    if enable_glow:
        _draw_glow_line(img_bgr, pts, color)
    cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def render_video(
    settings: RenderSettings,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> None:
    ensure_temp_dir(settings.output_path)
    gif_frames: List[np.ndarray] | None
    script_generate = None
    text_polylines: list[list[tuple[float, float]]] | None = None
    if settings.media_mode == "shapes":
        gif_frames = None
    elif settings.media_mode == "script":
        gif_frames = None
        script_generate = _load_script(settings.script_path)
    elif settings.media_mode == "text":
        gif_frames = None
        text_polylines = _text_to_polylines(
            settings.text_value,
            font_family=settings.text_font_family,
            scale=settings.text_scale,
        )
    else:
        gif_frames = _load_media_frames(
            settings.media_path,
            settings.width,
            settings.height,
            preserve_aspect=settings.preserve_aspect,
        )
        if not gif_frames:
            raise RuntimeError("No frames found in media")

    if settings.audio_mode == "osc":
        sr = 44100
        duration = max(0.5, float(settings.osc_duration))
        frame_hop = max(1, int(sr / settings.fps))
        total_frames = frame_count(duration, settings.fps)
        band_energies = np.zeros((total_frames, settings.bands), dtype=np.float32)
        for i in range(total_frames):
            t_sec = i / float(settings.fps)
            osc = _oscillator_value(settings, t_sec)
            band_energies[i, :] = osc
        analysis = AudioAnalysis(
            audio=np.zeros((2, int(duration * sr)), dtype=np.float32),
            sr=sr,
            frame_hop=frame_hop,
            band_energies=band_energies,
            duration=duration,
        )
    else:
        analysis = load_and_analyze(
            settings.audio_path,
            fps=settings.fps,
            bands=settings.bands,
        )

    total_frames = frame_count(analysis.duration, settings.fps)
    segment_size = max(256, analysis.frame_hop * 2)

    progress_log_path = init_progress_log(settings.output_path)

    tracker = ProgressTracker(
        render_total=total_frames,
        progress_cb=progress_cb,
        progress_log_path=progress_log_path,
    )

    frames: List[np.ndarray] = []
    height, width = settings.height, settings.width
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    barrel_map_x = None
    barrel_map_y = None
    if settings.barrel_amount != 0.0:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        nx = (grid_x - cx) / max(1.0, cx)
        ny = (grid_y - cy) / max(1.0, cy)
        r2 = nx * nx + ny * ny
        k = float(settings.barrel_amount)
        falloff = max(0.35, float(settings.barrel_falloff))
        factor = 1.0 + k * np.power(r2, falloff)
        map_x = (nx * factor) * cx + cx
        map_y = (ny * factor) * cy + cy
        barrel_map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
        barrel_map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
    smoothed_bands = None
    previous_frame = None
    for i in range(total_frames):
        if cancel_cb and cancel_cb():
            raise RenderCancelled("Render cancelled")
        center = i * analysis.frame_hop
        segment = _audio_segment(analysis.audio, center, segment_size)
        band_values = band_at_frame(analysis.band_energies, i)
        if settings.smoothing_enabled:
            if smoothed_bands is None:
                smoothed_bands = band_values.astype(np.float32)
            else:
                alpha = float(max(0.0, min(1.0, settings.smoothing_amount)))
                smoothed_bands = smoothed_bands + (band_values - smoothed_bands) * alpha
            band_values = smoothed_bands

        energy = float(band_values.mean()) if band_values.size else 0.0
        t_sec = i / float(settings.fps)
        osc = _oscillator_value(settings, t_sec)
        sig_displace_x = _mod_value(band_values, settings.mod_displace_x_band, osc, settings.osc_mix)
        sig_displace_y = _mod_value(band_values, settings.mod_displace_y_band, osc, settings.osc_mix)
        sig_thickness = _mod_value(band_values, settings.mod_thickness_band, osc, settings.osc_mix)
        sig_glow = _mod_value(band_values, settings.mod_glow_band, osc, settings.osc_mix)
        sig_threshold = _mod_value(band_values, settings.mod_threshold_band, osc, settings.osc_mix)
        sig_warp = _mod_value(band_values, settings.mod_warp_band, osc, settings.osc_mix)
        sig_warp_speed = _mod_value(band_values, settings.mod_warp_speed_band, osc, settings.osc_mix)
        sig_rotation = _mod_value(band_values, settings.mod_rotation_band, osc, settings.osc_mix)
        sig_rotation *= _rotation_direction(settings.mod_rotation_direction, t_sec)
        sig_flicker = _mod_value(band_values, settings.flicker_band, osc, settings.osc_mix)
        sig_hue = _mod_value(band_values, settings.hue_shift_band, osc, settings.osc_mix)
        sig_scan = _mod_value(band_values, settings.scanline_band, osc, settings.osc_mix)
        sig_jitter = _mod_value(band_values, settings.jitter_band, osc, settings.osc_mix)
        sig_dither = _mod_value(band_values, settings.dither_mod_band, osc, settings.osc_mix)

        if script_generate is not None:
            payload = _script_audio_payload(band_values, osc)
            polylines = script_generate(
                t_sec,
                payload,
                {
                    "width": settings.width,
                    "height": settings.height,
                    "fps": settings.fps,
                },
            )
            if not isinstance(polylines, list):
                raise RuntimeError("Script generate() must return a list of polylines")
            warp_phase = t_sec * (1.0 + settings.mod_warp_speed_amount * sig_warp_speed) * 6.0
            polylines = _warp_polylines(polylines, settings.mod_warp_amount * sig_warp, warp_phase)
            base = _make_script_frame(
                settings.width,
                settings.height,
                polylines,
                preserve_aspect=settings.preserve_aspect,
                decimate_step=max(1, int(settings.decimate_step)),
                jitter_amount=settings.jitter_amount * sig_jitter,
                jitter_axis=settings.jitter_axis,
                jitter_style=settings.jitter_style,
            )
        elif text_polylines is not None:
            text_rotation = 0.0
            if settings.mod_rotation_amount != 0.0:
                text_rotation = settings.mod_rotation_amount * sig_rotation
            polylines = _rotate_polylines(text_polylines, text_rotation)
            warp_phase = t_sec * (1.0 + settings.mod_warp_speed_amount * sig_warp_speed) * 6.0
            polylines = _warp_polylines(polylines, settings.mod_warp_amount * sig_warp, warp_phase)
            base = _make_script_frame(
                settings.width,
                settings.height,
                polylines,
                preserve_aspect=settings.preserve_aspect,
                decimate_step=max(1, int(settings.decimate_step)),
                jitter_amount=settings.jitter_amount * sig_jitter,
                jitter_axis=settings.jitter_axis,
                jitter_style=settings.jitter_style,
            )
        elif gif_frames is None:
            base_rotation = float(settings.shape_rotation)
            if settings.mod_rotation_amount != 0.0:
                base_rotation += settings.mod_rotation_amount * sig_rotation
            base = _make_shape_frame(
                width=settings.width,
                height=settings.height,
                shape_type=settings.shape_type,
                polygon_sides=settings.polygon_sides,
                shape_rotation=base_rotation,
                shape_size=settings.shape_size,
                star_points=settings.star_points,
                star_inner=settings.star_inner,
                rect_width=settings.rect_width,
                rect_height=settings.rect_height,
                ellipse_x=settings.ellipse_x,
                ellipse_y=settings.ellipse_y,
                spiral_turns=settings.spiral_turns,
                spiral_growth=settings.spiral_growth,
                lemniscate_scale=settings.lemniscate_scale,
                cardioid_scale=settings.cardioid_scale,
                clover_petals=settings.clover_petals,
                clover_scale=settings.clover_scale,
                superellipse_n=settings.superellipse_n,
                superellipse_scale=settings.superellipse_scale,
                decimate_step=max(1, int(settings.decimate_step)),
                jitter_amount=settings.jitter_amount * sig_jitter,
                jitter_axis=settings.jitter_axis,
                jitter_style=settings.jitter_style,
            )
        else:
            base = gif_frames[i % len(gif_frames)].copy()

        threshold = float(
            max(
                0.0,
                min(0.9, settings.edge_threshold - settings.mod_threshold_amount * sig_threshold),
            )
        )
        glow_strength = max(0.0, settings.glow_strength + settings.mod_glow_amount * sig_glow)
        thickness = max(1, int(round(1 + settings.mod_thickness_amount * sig_thickness)))
        edge_decimate_step = 1
        edge_jitter_amount = 0.0
        if settings.media_mode == "media":
            edge_decimate_step = max(1, int(settings.decimate_step))
            edge_jitter_amount = settings.jitter_amount * sig_jitter

        hue_shift = settings.hue_shift_amount * sig_hue
        shifted_color = _apply_hue_shift(settings.color_rgb, hue_shift)

        edge_rgb = _edge_oscilloscope(
            base,
            shifted_color,
            method=settings.edge_method,
            threshold=threshold,
            glow_strength=glow_strength,
            thickness=thickness,
            glow_radius=settings.glow_radius,
            glow_threshold=settings.glow_threshold,
            glow_blend=settings.glow_blend,
            decimate_step=edge_decimate_step,
            jitter_amount=edge_jitter_amount,
            jitter_axis=settings.jitter_axis,
            jitter_style=settings.jitter_style,
        )

        if settings.media_mode == "media" and settings.mod_rotation_amount != 0.0:
            media_rotation = settings.mod_rotation_amount * sig_rotation
            center = ((width - 1) * 0.5, (height - 1) * 0.5)
            rot_mat = cv2.getRotationMatrix2D(center, media_rotation, 1.0)
            edge_rgb = cv2.warpAffine(
                edge_rgb,
                rot_mat,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        phase = i / float(settings.fps)
        warp_amount = settings.mod_warp_amount * sig_warp
        warp_speed = 1.0 + settings.mod_warp_speed_amount * sig_warp_speed
        displace_x = settings.mod_displace_x_amount * sig_displace_x
        displace_y = settings.mod_displace_y_amount * sig_displace_y
        if warp_amount != 0.0 or displace_x != 0.0 or displace_y != 0.0:
            freq_x = 2 * np.pi / max(1.0, height / 3.0)
            freq_y = 2 * np.pi / max(1.0, width / 3.0)
            warp_x = np.sin(grid_y * freq_x + phase * warp_speed) * warp_amount
            warp_y = np.sin(grid_x * freq_y + phase * (warp_speed * 1.1)) * warp_amount

            jitter_x = np.sin(grid_y * (freq_x * 1.7) + phase * (warp_speed * 1.7)) * displace_x
            jitter_y = np.cos(grid_x * (freq_y * 1.3) + phase * (warp_speed * 1.3)) * displace_y

            map_x = np.clip(grid_x + warp_x + jitter_x, 0, width - 1)
            map_y = np.clip(grid_y + warp_y + jitter_y, 0, height - 1)
            edge_rgb = cv2.remap(edge_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        img_bgr = cv2.cvtColor(edge_rgb, cv2.COLOR_RGB2BGR)

        if settings.edge_mode != "edge_only" and settings.enable_line:
            waveform = _make_waveform(segment, settings.width)
            _draw_waveform(
                img_bgr,
                waveform,
                band_values,
                shifted_color,
                enable_glow=settings.enable_glow,
                decimate_step=max(1, int(settings.decimate_step)),
                jitter_amount=settings.jitter_amount * sig_jitter,
                jitter_axis=settings.jitter_axis,
                jitter_style=settings.jitter_style,
            )

        if settings.edge_mode != "edge_only" and settings.enable_lissajous:
            lissajous = _make_lissajous(segment, 512)
            _draw_lissajous(
                img_bgr,
                lissajous,
                band_values,
                shifted_color,
                enable_glow=settings.enable_glow,
                decimate_step=max(1, int(settings.decimate_step)),
                jitter_amount=settings.jitter_amount * sig_jitter,
                jitter_axis=settings.jitter_axis,
                jitter_style=settings.jitter_style,
            )

        if settings.scanline_amount > 0.0:
            amount = settings.scanline_amount * sig_scan
            img_bgr = _apply_scanlines(
                img_bgr,
                amount=amount,
                speed=settings.scanline_speed,
                t=t_sec,
                thickness=settings.scanline_thickness,
                spacing=settings.scanline_spacing,
                style=settings.scanline_style,
            )

        if settings.flicker_amount > 0.0:
            factor = _flicker_factor(
                t=t_sec,
                amount=settings.flicker_amount,
                signal=sig_flicker,
                speed=settings.flicker_speed,
                floor=settings.flicker_floor,
                style=settings.flicker_style,
            )
            img_bgr = np.clip(img_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if settings.trail_strength > 0.0:
            img_bgr = _apply_trail(
                img_bgr,
                previous_frame,
                strength=settings.trail_strength,
                decay=settings.trail_decay,
                blend=settings.trail_blend,
            )

        if settings.bloom_amount > 0.0:
            img_bgr = _apply_bloom(
                img_bgr,
                amount=settings.bloom_amount,
                radius=settings.bloom_radius,
                threshold=settings.bloom_threshold,
            )

        if settings.vignette_amount > 0.0:
            img_bgr = _apply_vignette(
                img_bgr,
                amount=settings.vignette_amount,
                power=settings.vignette_power,
                grid_x=grid_x,
                grid_y=grid_y,
            )

        if settings.phosphor_amount > 0.0:
            img_bgr = _apply_phosphor_mask(
                img_bgr,
                amount=settings.phosphor_amount,
                style=settings.phosphor_style,
                width=settings.phosphor_width,
            )

        if settings.bleed_amount > 0.0:
            img_bgr = _apply_color_bleed_advanced(
                img_bgr,
                amount=settings.bleed_amount,
                radius=settings.bleed_radius,
                direction=settings.bleed_direction,
            )

        if settings.chroma_shift_x != 0.0 or settings.chroma_shift_y != 0.0:
            img_bgr = _apply_chromatic_aberration(
                img_bgr,
                shift_x=settings.chroma_shift_x,
                shift_y=settings.chroma_shift_y,
            )

        if settings.barrel_amount != 0.0:
            img_bgr = _apply_barrel_distortion(img_bgr, barrel_map_x, barrel_map_y)

        if settings.h_jitter_amount > 0.0:
            img_bgr = _apply_horizontal_jitter(
                img_bgr,
                amount=settings.h_jitter_amount,
                speed=settings.h_jitter_speed,
                t=t_sec,
                grid_x=grid_x,
                grid_y=grid_y,
            )

        if settings.v_roll_amount > 0.0:
            img_bgr = _apply_vertical_roll(
                img_bgr,
                amount=settings.v_roll_amount,
                speed=settings.v_roll_speed,
                t=t_sec,
            )

        if settings.noise_amount > 0.0:
            img_bgr = _apply_noise(
                img_bgr,
                amount=settings.noise_amount,
                mode=settings.noise_mode,
                grain=settings.noise_grain,
            )

        dither_amount = max(0.0, min(1.0, settings.dither_amount + settings.dither_mod_amount * sig_dither))
        if dither_amount > 0.0:
            img_bgr = _apply_dither(
                img_bgr,
                amount=dither_amount,
                mode=settings.dither_mode,
                levels=settings.dither_levels,
            )

        previous_frame = img_bgr.copy()

        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        tracker.update_render(i + 1)

    if cancel_cb and cancel_cb():
        raise RenderCancelled("Render cancelled")

    clip = ImageSequenceClip(frames, fps=settings.fps)
    logger = MoviepyLogger(tracker)
    if settings.audio_mode == "osc":
        clip.write_videofile(
            settings.output_path,
            codec="libx264",
            audio=False,
            fps=settings.fps,
            threads=4,
            logger=logger,
        )
    else:
        temp_audio_dir = temp_dir()
        temp_audiofile = str(temp_audio_dir / f"{Path(settings.output_path).stem}_TEMP_MPY_audio.m4a")
        try:
            os.remove(temp_audiofile)
        except FileNotFoundError:
            pass
        clip = clip.with_audio(AudioFileClip(settings.audio_path))
        clip.write_videofile(
            settings.output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audiofile,
            temp_audiofile_path=str(temp_audio_dir),
            remove_temp=True,
            fps=settings.fps,
            threads=4,
            logger=logger,
        )
    tracker.finish()
