from __future__ import annotations

import cv2
import numpy as np

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

_BAYER_4 = (1.0 / 16.0) * np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
)


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
    falloff = np.clip(1.0 - amount * (radius**power), 0.0, 1.0)
    out = img_bgr.astype(np.float32)
    out *= falloff[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_phosphor_mask(
    img_bgr: np.ndarray,
    *,
    amount: float,
    style: str = "rgb",
    width: int = 3,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, frame_width = img_bgr.shape[:2]
    stripe_width = max(1, int(width))
    x = np.arange(frame_width, dtype=np.int32)
    phase = (x // stripe_width)
    mask = np.ones((height, frame_width, 3), dtype=np.float32)
    if style == "grille":
        stripe = phase % 2
        brightness = np.where(stripe == 0, 1.0, 0.62).astype(np.float32)
        mask *= brightness[None, :, None]
    elif style == "slot":
        stripe = phase % 3
        mask[..., 2] = np.where(stripe == 0, 1.0, 0.72)
        mask[..., 1] = np.where(stripe == 1, 1.0, 0.72)
        mask[..., 0] = np.where(stripe == 2, 1.0, 0.72)
        y = np.arange(height, dtype=np.int32)[:, None]
        rows = ((y // max(1, stripe_width)) % 3 == 1).astype(np.float32)
        mask *= (0.82 + 0.18 * rows)[..., None]
    else:
        stripe = phase % 3
        mask[..., 2] = np.where(stripe == 0, 1.0, 0.6)
        mask[..., 1] = np.where(stripe == 1, 1.0, 0.6)
        mask[..., 0] = np.where(stripe == 2, 1.0, 0.6)
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


def _flicker_factor(
    *,
    t: float,
    amount: float,
    signal: float,
    speed: float,
    floor: float,
    style: str,
) -> float:
    if amount <= 0.0:
        return 1.0
    speed = max(0.05, float(speed))
    floor = max(0.2, min(1.2, float(floor)))
    ceiling = 1.0 + amount * (0.18 + 0.35 * max(0.0, float(signal)))
    if style == "square":
        wave = 1.0 if np.sin(t * speed * 2.0 * np.pi) >= 0.0 else 0.0
        factor = floor + (ceiling - floor) * wave
    elif style == "sine":
        wave = 0.5 + 0.5 * np.sin(t * speed * 2.0 * np.pi)
        factor = floor + (ceiling - floor) * wave
    else:
        wave = (
            0.55
            + 0.25 * np.sin(t * speed * 9.7)
            + 0.14 * np.sin(t * speed * 23.0 + 1.3)
            + 0.06 * np.sin(t * speed * 57.0 + 0.4)
        )
        wave = max(0.0, min(1.0, wave))
        factor = floor + (ceiling - floor) * wave
    return max(0.2, min(1.8, float(factor)))


def _apply_scanlines(
    img_bgr: np.ndarray,
    *,
    amount: float,
    speed: float,
    t: float,
    thickness: int,
    spacing: int,
    style: str,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, width = img_bgr.shape[:2]
    spacing = max(2, int(spacing))
    thickness = max(1, int(thickness))
    out = img_bgr.astype(np.float32).copy()
    phase = t * speed * 12.0
    for y in range(0, height, spacing):
        wave = 0.5 + 0.5 * np.sin((y * 0.12) + phase)
        strength = amount * (0.25 + 0.75 * wave)
        y_end = min(height, y + thickness)
        if style == "light":
            out[y:y_end] = out[y:y_end] + (255.0 - out[y:y_end]) * (0.45 * strength)
        else:
            out[y:y_end] = out[y:y_end] * (1.0 - 0.7 * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_noise(
    img_bgr: np.ndarray,
    *,
    amount: float,
    mode: str = "rgb",
    grain: float = 1.0,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    height, width = img_bgr.shape[:2]
    grain = max(1.0, float(grain))
    sample_w = max(1, int(round(width / grain)))
    sample_h = max(1, int(round(height / grain)))
    scale = amount * 30.0
    if mode == "mono":
        base_noise = np.random.randn(sample_h, sample_w, 1).astype(np.float32) * scale
        noise = np.repeat(base_noise, 3, axis=2)
    else:
        noise = np.random.randn(sample_h, sample_w, 3).astype(np.float32) * scale
    if sample_w != width or sample_h != height:
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
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


def _apply_color_bleed_advanced(
    img_bgr: np.ndarray,
    *,
    amount: float,
    radius: float,
    direction: str,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    k = int(1 + radius * 20)
    if k % 2 == 0:
        k += 1
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    if direction == "vertical":
        kernel = (1, k)
    elif direction == "both":
        kernel = (k, k)
    else:
        kernel = (k, 1)
    cr_blur = cv2.GaussianBlur(cr, kernel, sigmaX=0)
    cb_blur = cv2.GaussianBlur(cb, kernel, sigmaX=0)
    cr = cv2.addWeighted(cr, 1.0 - amount, cr_blur, amount, 0)
    cb = cv2.addWeighted(cb, 1.0 - amount, cb_blur, amount, 0)
    merged = cv2.merge([y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _apply_trail(
    img_bgr: np.ndarray,
    previous_frame: np.ndarray | None,
    *,
    strength: float,
    decay: float,
    blend: str,
) -> np.ndarray:
    if previous_frame is None or strength <= 0.0:
        return img_bgr
    strength = max(0.0, min(1.0, float(strength)))
    decay = max(0.0, min(1.0, float(decay)))
    prev = previous_frame.astype(np.float32) * decay
    curr = img_bgr.astype(np.float32)
    if blend == "add":
        out = curr + prev * strength
    elif blend == "lighten":
        out = np.maximum(curr, prev * strength + curr * (1.0 - strength))
    else:
        out = curr * (1.0 - strength) + prev * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _ordered_threshold_map(height: int, width: int, mode: str) -> np.ndarray:
    matrix = _BAYER_8 if mode == "bayer" else _BAYER_4
    tile_h, tile_w = matrix.shape
    return np.tile(matrix, (height // tile_h + 1, width // tile_w + 1))[:height, :width]


def _quantize_levels(img: np.ndarray, levels: int) -> np.ndarray:
    steps = max(2, int(levels)) - 1
    return np.clip(np.round(img * steps) / steps, 0.0, 1.0)


def _floyd_steinberg_dither(img: np.ndarray, levels: int) -> np.ndarray:
    steps = max(2, int(levels)) - 1
    work = img.astype(np.float32).copy()
    height, width = work.shape[:2]
    for y in range(height):
        for x in range(width):
            old = work[y, x].copy()
            new = np.clip(np.round(old * steps) / steps, 0.0, 1.0)
            work[y, x] = new
            err = old - new
            if x + 1 < width:
                work[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < height:
                if x > 0:
                    work[y + 1, x - 1] += err * (3.0 / 16.0)
                work[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < width:
                    work[y + 1, x + 1] += err * (1.0 / 16.0)
    return np.clip(work, 0.0, 1.0)


def _apply_dither(
    img_bgr: np.ndarray,
    *,
    amount: float,
    mode: str = "bayer",
    levels: int = 4,
) -> np.ndarray:
    if amount <= 0.0:
        return img_bgr
    img = img_bgr.astype(np.float32) / 255.0
    levels = max(2, min(32, int(levels)))
    amount = max(0.0, min(1.0, float(amount)))

    if mode == "diffusion":
        quantized = _floyd_steinberg_dither(img, levels)
        mixed = img * (1.0 - amount) + quantized * amount
        return (np.clip(mixed, 0.0, 1.0) * 255.0).astype(np.uint8)

    height, width = img.shape[:2]
    threshold_map = _ordered_threshold_map(height, width, mode if mode in {"bayer", "ordered"} else "bayer")
    spread = amount / max(1, levels - 1)
    biased = np.clip(img + (threshold_map[..., None] - 0.5) * spread, 0.0, 1.0)
    quantized = _quantize_levels(biased, levels)
    mixed = img * (1.0 - amount) + quantized * amount
    return (np.clip(mixed, 0.0, 1.0) * 255.0).astype(np.uint8)

__all__ = [
    "_apply_bloom",
    "_apply_vignette",
    "_apply_phosphor_mask",
    "_apply_chromatic_aberration",
    "_apply_barrel_distortion",
    "_flicker_factor",
    "_apply_scanlines",
    "_apply_noise",
    "_apply_horizontal_jitter",
    "_apply_vertical_roll",
    "_apply_color_bleed",
    "_apply_dither",
]
