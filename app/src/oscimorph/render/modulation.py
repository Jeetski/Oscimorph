from __future__ import annotations

import colorsys

import numpy as np

from .settings import RenderSettings


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

__all__ = [
    "_apply_hue_shift",
    "_mod_value",
    "_oscillator_value",
    "_rotation_direction",
    "_script_audio_payload",
]
