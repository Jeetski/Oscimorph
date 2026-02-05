from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import time
from typing import Callable, List
import colorsys

import cv2
import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageSequenceClip
from proglog import ProgressBarLogger
from PySide6.QtGui import QFont, QPainterPath

from .audio import band_at_frame, frame_count, load_and_analyze

_BAYER_8 = (1.0 / 64.0) * np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float32,
)


@dataclass
class RenderSettings:
    media_path: str
    audio_path: str
    output_path: str
    audio_mode: str = "file"  # "file" or "osc"
    osc_duration: float = 10.0
    width: int = 1280
    height: int = 720
    fps: int = 30
    enable_line: bool = True
    enable_lissajous: bool = True
    enable_glow: bool = True
    bands: int = 5
    preserve_aspect: bool = True
    edge_mode: str = "edge_overlay"
    edge_method: str = "sobel"
    edge_threshold: float = 0.08
    glow_strength: float = 0.85
    color_rgb: tuple[int, int, int] = (0, 255, 204)
    smoothing_enabled: bool = False
    smoothing_amount: float = 0.2
    mod_displace_x_amount: float = 6.0
    mod_displace_x_band: str = "low"
    mod_displace_y_amount: float = 6.0
    mod_displace_y_band: str = "mid"
    mod_thickness_amount: float = 3.0
    mod_thickness_band: str = "all"
    mod_glow_amount: float = 1.0
    mod_glow_band: str = "all"
    mod_threshold_amount: float = 0.05
    mod_threshold_band: str = "all"
    mod_warp_amount: float = 8.0
    mod_warp_band: str = "high"
    mod_warp_speed_amount: float = 2.0
    mod_warp_speed_band: str = "mid"
    media_mode: str = "media"  # "media", "shapes", or "script"
    shape_type: str = "ring"  # "ring" or "polygon"
    polygon_sides: int = 5
    shape_rotation: float = 0.0
    shape_size: float = 1.0
    star_points: int = 5
    star_inner: float = 0.45
    rect_width: float = 1.0
    rect_height: float = 0.6
    ellipse_x: float = 1.2
    ellipse_y: float = 0.8
    spiral_turns: float = 3.5
    spiral_growth: float = 1.0
    lemniscate_scale: float = 1.4
    cardioid_scale: float = 1.0
    clover_petals: int = 4
    clover_scale: float = 1.2
    superellipse_n: float = 2.6
    superellipse_scale: float = 1.1
    mod_rotation_amount: float = 0.0
    mod_rotation_band: str = "all"
    mod_rotation_direction: str = "cw"
    script_path: str = ""
    trail_strength: float = 0.0
    glow_radius: float = 2.2
    flicker_amount: float = 0.0
    flicker_band: str = "all"
    hue_shift_amount: float = 0.0
    hue_shift_band: str = "all"
    scanline_amount: float = 0.0
    scanline_band: str = "all"
    scanline_speed: float = 1.0
    decimate_step: int = 1
    jitter_amount: float = 0.0
    jitter_band: str = "all"
    osc_waveform: str = "sine"
    osc_frequency: float = 0.5
    osc_depth: float = 1.0
    osc_mix: float = 0.0
    text_value: str = "OSCIMORPH"
    text_scale: float = 1.0
    text_font_family: str = ""
    dither_amount: float = 0.2
    phosphor_amount: float = 0.35
    bloom_amount: float = 0.35
    bloom_radius: float = 2.5
    bloom_threshold: float = 0.6
    vignette_amount: float = 0.35
    vignette_power: float = 1.8
    chroma_shift_x: float = 1.0
    chroma_shift_y: float = 0.5
    barrel_amount: float = 0.12
    noise_amount: float = 0.15
    h_jitter_amount: float = 2.0
    h_jitter_speed: float = 2.0
    v_roll_amount: float = 8.0
    v_roll_speed: float = 0.35
    bleed_amount: float = 0.25


class RenderCancelled(RuntimeError):
    pass


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
        ).astype(np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    elif shape_type == "ellipse":
        cv2.ellipse(
            canvas,
            center,
            (int(radius * ellipse_x), int(radius * ellipse_y)),
            shape_rotation,
            0,
            360,
            color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts.astype(np.int32)], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
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
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    else:
        cv2.ellipse(
            canvas,
            center,
            (radius, radius),
            shape_rotation,
            0,
            360,
            color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    return canvas


def _load_script(path: str):
    if not os.path.exists(path):
        raise RuntimeError("Script file not found")
    namespace: dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as handle:
        code = handle.read()
    safe_globals = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "random": __import__("random"),
        "np": np,
    }
    exec(compile(code, path, "exec"), safe_globals, namespace)  # noqa: S102 - user script
    generate = namespace.get("generate") or safe_globals.get("generate")
    if not callable(generate):
        raise RuntimeError("Script must define a callable generate(t, audio, settings)")
    return generate


def _text_to_polylines(text: str, *, font_family: str, scale: float) -> list[list[tuple[float, float]]]:
    if not text.strip():
        raise RuntimeError("Text is empty")
    font = QFont(font_family)
    if not font_family:
        font = QFont()
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setPointSizeF(100.0)
    path = QPainterPath()
    path.addText(0, 0, font, text)
    polygons = path.toSubpathPolygons()
    if not polygons:
        raise RuntimeError("Unable to extract text outlines")

    points: list[tuple[float, float]] = []
    polylines: list[list[tuple[float, float]]] = []
    for poly in polygons:
        line = [(pt.x(), pt.y()) for pt in poly]
        if line:
            polylines.append(line)
            points.extend(line)
    if not points:
        raise RuntimeError("Unable to extract text outlines")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise RuntimeError("Text outlines have zero area")
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    normalize = 2.0 / max(width, height)
    scale = float(max(0.05, min(5.0, scale)))

    normalized: list[list[tuple[float, float]]] = []
    for line in polylines:
        normalized.append(
            [((x - center_x) * normalize * scale, (y - center_y) * normalize * scale) for x, y in line]
        )
    return normalized


def _script_audio_payload(bands: np.ndarray, osc: float) -> dict[str, float]:
    if bands.size < 5:
        return {
            "subs": 0.0,
            "lows": 0.0,
            "low_mids": 0.0,
            "high_mids": 0.0,
            "highs": 0.0,
            "all": float(bands.mean()) if bands.size else 0.0,
            "osc": osc,
        }
    return {
        "subs": float(bands[0]),
        "lows": float(bands[1]),
        "low_mids": float(bands[2]),
        "high_mids": float(bands[3]),
        "highs": float(bands[4]),
        "all": float(bands.mean()),
        "osc": osc,
    }


def _make_script_frame(
    width: int,
    height: int,
    polylines: list[list[tuple[float, float]]],
    *,
    preserve_aspect: bool,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if not polylines:
        return canvas
    if preserve_aspect:
        scale = min(width, height) * 0.5
        offset_x = width * 0.5
        offset_y = height * 0.5
    for line in polylines:
        if not line:
            continue
        if preserve_aspect:
            pts = np.array(
                [
                    (
                        offset_x + x * scale,
                        offset_y + y * scale,
                    )
                    for x, y in line
                ],
                dtype=np.int32,
            )
        else:
            pts = np.array(
                [
                    (
                        (x * 0.5 + 0.5) * (width - 1),
                        (y * 0.5 + 0.5) * (height - 1),
                    )
                    for x, y in line
                ],
                dtype=np.int32,
            )
        cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
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


def _apply_hue_shift(color_rgb: tuple[int, int, int], shift_degrees: float) -> tuple[int, int, int]:
    r, g, b = [c / 255.0 for c in color_rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + (shift_degrees / 360.0)) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def _select_band_value(bands: np.ndarray, selector: str) -> float:
    if bands.size == 0:
        return 0.0
    if selector == "osc":
        return 0.0
    if selector == "all":
        return float(bands.mean())
    if selector == "low":
        return float(bands[0])
    if selector == "mid":
        return float(bands[bands.size // 2])
    if selector == "high":
        return float(bands[-1])
    if selector.startswith("band:"):
        try:
            index = int(selector.split(":", 1)[1])
        except ValueError:
            return float(bands.mean())
        index = max(0, min(bands.size - 1, index))
        return float(bands[index])
    return float(bands.mean())


def _oscillator_value(settings: RenderSettings, t: float) -> float:
    phase = t * 2 * np.pi * settings.osc_frequency
    if settings.osc_waveform == "triangle":
        raw = 2.0 * np.abs(2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
    elif settings.osc_waveform == "square":
        raw = 1.0 if np.sin(phase) >= 0 else -1.0
    elif settings.osc_waveform == "saw":
        raw = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    else:
        raw = np.sin(phase)
    value = raw * float(settings.osc_depth)
    return float(max(-1.0, min(1.0, value)) * 0.5 + 0.5)


def _mod_value(bands: np.ndarray, selector: str, osc: float, osc_mix: float) -> float:
    if selector == "osc":
        return osc
    audio = _select_band_value(bands, selector)
    if osc_mix <= 0.0:
        return audio
    mix = max(0.0, min(1.0, osc_mix))
    return audio * (1.0 - mix) + osc * mix


def _rotation_direction(direction: str, t: float) -> float:
    if direction == "ccw":
        return -1.0
    if direction == "alternate":
        return 1.0 if np.sin(2 * np.pi * t) >= 0 else -1.0
    return 1.0


def _edge_oscilloscope(
    frame_rgb: np.ndarray,
    color_rgb: tuple[int, int, int],
    *,
    method: str,
    threshold: float,
    glow_strength: float,
    thickness: int,
    glow_radius: float,
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

    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        mag = cv2.dilate(mag, kernel, iterations=1)

    core = mag
    glow = cv2.GaussianBlur(mag, (0, 0), sigmaX=max(0.5, glow_radius))
    glow = np.clip(glow, 0, 1)

    intensity = np.clip(core * 1.0 + glow * glow_strength, 0, 1)
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
        ys = ys + jitter
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
) -> None:
    height, width = img_bgr.shape[:2]
    points = lissajous.copy()
    if decimate_step > 1:
        points = points[::decimate_step]
    points[:, 0] = (points[:, 0] * 0.5 + 0.5) * (width - 1)
    points[:, 1] = (points[:, 1] * 0.5 + 0.5) * (height - 1)
    if jitter_amount > 0.0:
        jitter = (np.random.rand(points.shape[0], 2) - 0.5) * 2.0 * jitter_amount
        points = points + jitter
    pts = points.astype(np.int32)
    energy = float(band_values.mean())
    color = _color_with_energy(color_rgb, energy)

    if enable_glow:
        _draw_glow_line(img_bgr, pts, color)
    cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def _apply_bloom(
    img_bgr: np.ndarray,
    *,
    amount: float,
    radius: float,
    threshold: float,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    img = img_bgr.astype(np.float32) / 255.0
    luma = (0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0])
    mask = np.clip((luma - threshold) / max(1e-6, 1 - threshold), 0, 1)
    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.5, radius))
    blur = blur[..., None]
    img = np.clip(img + blur * amount, 0, 1)
    return (img * 255.0).astype(np.uint8)


def _apply_vignette(
    img_bgr: np.ndarray,
    *,
    amount: float,
    power: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, width = img_bgr.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    nx = (grid_x - cx) / max(1.0, cx)
    ny = (grid_y - cy) / max(1.0, cy)
    radius = np.sqrt(nx * nx + ny * ny)
    falloff = np.clip(1.0 - amount * (radius ** power), 0.0, 1.0)
    out = img_bgr.astype(np.float32)
    out *= falloff[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_phosphor_mask(img_bgr: np.ndarray, *, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, width = img_bgr.shape[:2]
    x = np.arange(width, dtype=np.int32)
    stripe = x % 3
    mask = np.ones((height, width, 3), dtype=np.float32)
    mask[..., 2] = np.where(stripe == 0, 1.0, 0.6)  # R
    mask[..., 1] = np.where(stripe == 1, 1.0, 0.6)  # G
    mask[..., 0] = np.where(stripe == 2, 1.0, 0.6)  # B
    base = img_bgr.astype(np.float32) / 255.0
    masked = np.clip(base * mask, 0, 1)
    out = base * (1.0 - amount) + masked * amount
    return (out * 255.0).astype(np.uint8)


def _apply_chromatic_aberration(
    img_bgr: np.ndarray,
    *,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    if abs(shift_x) < 1e-3 and abs(shift_y) < 1e-3:
        return img_bgr
    height, width = img_bgr.shape[:2]
    b, g, r = cv2.split(img_bgr)
    mat = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    r = cv2.warpAffine(r, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mat = np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]], dtype=np.float32)
    b = cv2.warpAffine(b, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.merge([b, g, r])


def _apply_barrel_distortion(
    img_bgr: np.ndarray,
    map_x: np.ndarray | None,
    map_y: np.ndarray | None,
) -> np.ndarray:
    if map_x is None or map_y is None:
        return img_bgr
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _apply_noise(img_bgr: np.ndarray, *, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    scale = amount * 30.0
    noise = np.random.randn(*img_bgr.shape).astype(np.float32) * scale
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_horizontal_jitter(
    img_bgr: np.ndarray,
    *,
    amount: float,
    speed: float,
    t: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, width = img_bgr.shape[:2]
    freq = 2 * np.pi / max(1.0, height / 6.0)
    offset = np.sin(grid_y * freq + t * speed * 2 * np.pi) * amount
    map_x = np.clip(grid_x + offset, 0, width - 1)
    map_y = grid_y
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _apply_vertical_roll(
    img_bgr: np.ndarray,
    *,
    amount: float,
    speed: float,
    t: float,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height = img_bgr.shape[0]
    shift = int((t * speed * amount) % height)
    if shift == 0:
        return img_bgr
    return np.roll(img_bgr, shift, axis=0)


def _apply_color_bleed(img_bgr: np.ndarray, *, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    k = int(1 + amount * 20)
    if k % 2 == 0:
        k += 1
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.GaussianBlur(cr, (k, 1), sigmaX=0)
    cb = cv2.GaussianBlur(cb, (k, 1), sigmaX=0)
    merged = cv2.merge([y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _apply_dither(img_bgr: np.ndarray, *, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, width = img_bgr.shape[:2]
    tiled = np.tile(_BAYER_8, (height // 8 + 1, width // 8 + 1))[:height, :width]
    img = img_bgr.astype(np.float32) / 255.0
    dither = (tiled - 0.5) * amount
    img = np.clip(img + dither[..., None], 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def render_video(
    settings: RenderSettings,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> None:
    _ensure_temp_dir(settings.output_path)
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

    progress_log_path = _init_progress_log(settings.output_path)

    tracker = _ProgressTracker(
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
        factor = 1.0 + k * r2
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
            base = _make_script_frame(
                settings.width,
                settings.height,
                polylines,
                preserve_aspect=settings.preserve_aspect,
            )
        elif text_polylines is not None:
            base = _make_script_frame(
                settings.width,
                settings.height,
                text_polylines,
                preserve_aspect=settings.preserve_aspect,
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
        )

        if settings.edge_mode == "edge_only":
            phase = i / float(settings.fps)
            warp_amount = settings.mod_warp_amount * sig_warp
            warp_speed = 1.0 + settings.mod_warp_speed_amount * sig_warp_speed
            freq_x = 2 * np.pi / max(1.0, height / 3.0)
            freq_y = 2 * np.pi / max(1.0, width / 3.0)
            warp_x = np.sin(grid_y * freq_x + phase * warp_speed) * warp_amount
            warp_y = np.sin(grid_x * freq_y + phase * (warp_speed * 1.1)) * warp_amount

            displace_x = settings.mod_displace_x_amount * sig_displace_x
            displace_y = settings.mod_displace_y_amount * sig_displace_y
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
            )

        if settings.scanline_amount > 0.0:
            phase = i / float(settings.fps)
            amount = settings.scanline_amount * sig_scan
            freq = 2 * np.pi / max(1.0, height / 6.0)
            offset = np.sin(grid_y * freq + phase * settings.scanline_speed) * amount
            map_x = np.clip(grid_x + offset, 0, width - 1)
            map_y = grid_y
            img_bgr = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        if settings.flicker_amount > 0.0:
            jitter = (np.random.rand() - 0.5) * 2.0
            factor = 1.0 + settings.flicker_amount * (0.4 * sig_flicker + 0.2 * jitter)
            factor = max(0.5, min(1.5, factor))
            img_bgr = np.clip(img_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if settings.trail_strength > 0.0 and previous_frame is not None:
            trail = max(0.0, min(1.0, settings.trail_strength))
            img_bgr = cv2.addWeighted(img_bgr, 1 - trail, previous_frame, trail, 0)

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
            img_bgr = _apply_phosphor_mask(img_bgr, amount=settings.phosphor_amount)

        if settings.bleed_amount > 0.0:
            img_bgr = _apply_color_bleed(img_bgr, amount=settings.bleed_amount)

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
            img_bgr = _apply_noise(img_bgr, amount=settings.noise_amount)

        if settings.dither_amount > 0.0:
            img_bgr = _apply_dither(img_bgr, amount=settings.dither_amount)

        previous_frame = img_bgr.copy()

        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        tracker.update_render(i + 1)

    if cancel_cb and cancel_cb():
        raise RenderCancelled("Render cancelled")

    clip = ImageSequenceClip(frames, fps=settings.fps)
    logger = _MoviepyLogger(tracker)
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
        clip = clip.with_audio(AudioFileClip(settings.audio_path))
        clip.write_videofile(
            settings.output_path,
            codec="libx264",
            audio_codec="aac",
            fps=settings.fps,
            threads=4,
            logger=logger,
        )
    tracker.finish()


class _ProgressTracker:
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


class _MoviepyLogger(ProgressBarLogger):
    def __init__(self, tracker: _ProgressTracker) -> None:
        super().__init__(min_time_interval=0.2)
        self._tracker = tracker

    def bars_callback(self, bar, attr, value, old_value=None):  # noqa: ANN001
        self._tracker.update_bar(bar, attr, value)


def _init_progress_log(output_path: str) -> str | None:
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


def _ensure_temp_dir(output_path: str) -> None:
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
