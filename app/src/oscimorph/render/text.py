from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtGui import QFont, QPainterPath


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
    exec(compile(code, path, "exec"), safe_globals, namespace)  # noqa: S102
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


def _rotate_polylines(
    polylines: list[list[tuple[float, float]]],
    degrees: float,
) -> list[list[tuple[float, float]]]:
    if abs(degrees) < 1e-3:
        return polylines
    rot = np.deg2rad(degrees)
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    rotated: list[list[tuple[float, float]]] = []
    for line in polylines:
        rotated.append([(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for x, y in line])
    return rotated


def _warp_polylines(
    polylines: list[list[tuple[float, float]]],
    amount: float,
    phase: float,
) -> list[list[tuple[float, float]]]:
    warp = max(0.0, amount) * 0.003
    if warp <= 0.0:
        return polylines
    warped: list[list[tuple[float, float]]] = []
    for line in polylines:
        out: list[tuple[float, float]] = []
        for x, y in line:
            angle = np.arctan2(y, x)
            radius = np.sqrt(x * x + y * y)
            radius *= 1.0 + warp * np.sin(angle * 3.0 + phase)
            out.append((np.cos(angle) * radius, np.sin(angle) * radius))
        warped.append(out)
    return warped


def _make_script_frame(
    width: int,
    height: int,
    polylines: list[list[tuple[float, float]]],
    *,
    preserve_aspect: bool,
    decimate_step: int = 1,
    jitter_amount: float = 0.0,
    jitter_axis: str = "xy",
    jitter_style: str = "random",
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if not polylines:
        return canvas
    step = max(1, int(decimate_step))
    if preserve_aspect:
        scale = min(width, height) * 0.5
        offset_x = width * 0.5
        offset_y = height * 0.5
    for line in polylines:
        if not line:
            continue
        points = line[::step] if step > 1 else line
        if not points:
            continue
        transformed: list[tuple[float, float]] = []
        for x, y in points:
            if jitter_amount > 0.0:
                jx = (np.random.rand() - 0.5) * 2.0 * jitter_amount
                jy = (np.random.rand() - 0.5) * 2.0 * jitter_amount
                if jitter_style == "stepped":
                    jx = round(jx)
                    jy = round(jy)
                if jitter_axis in {"xy", "x"}:
                    x += jx
                if jitter_axis in {"xy", "y"}:
                    y += jy
            transformed.append((x, y))
        if preserve_aspect:
            pts = np.array(
                [(offset_x + x * scale, offset_y + y * scale) for x, y in transformed],
                dtype=np.int32,
            )
        else:
            pts = np.array(
                [((x * 0.5 + 0.5) * (width - 1), (y * 0.5 + 0.5) * (height - 1)) for x, y in transformed],
                dtype=np.int32,
            )
        cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return canvas

__all__ = [
    "_load_script",
    "_make_script_frame",
    "_rotate_polylines",
    "_text_to_polylines",
    "_warp_polylines",
]
