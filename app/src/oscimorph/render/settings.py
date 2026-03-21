from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderSettings:
    media_path: str
    audio_path: str
    output_path: str
    audio_mode: str = "file"
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
    media_mode: str = "media"
    shape_type: str = "ring"
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
    trail_decay: float = 1.0
    trail_blend: str = "mix"
    glow_radius: float = 2.2
    glow_threshold: float = 0.0
    glow_blend: str = "add"
    flicker_amount: float = 0.0
    flicker_band: str = "all"
    flicker_style: str = "random"
    flicker_speed: float = 1.0
    flicker_floor: float = 0.75
    hue_shift_amount: float = 0.0
    hue_shift_band: str = "all"
    scanline_amount: float = 0.0
    scanline_band: str = "all"
    scanline_speed: float = 1.0
    scanline_thickness: int = 1
    scanline_spacing: int = 3
    scanline_style: str = "dark"
    decimate_step: int = 1
    jitter_amount: float = 0.0
    jitter_band: str = "all"
    jitter_axis: str = "xy"
    jitter_style: str = "random"
    osc_waveform: str = "sine"
    osc_frequency: float = 0.5
    osc_depth: float = 1.0
    osc_mix: float = 0.0
    text_value: str = "OSCIMORPH"
    text_scale: float = 1.0
    text_font_family: str = ""
    dither_amount: float = 0.2
    dither_mod_amount: float = 0.0
    dither_mod_band: str = "all"
    dither_mode: str = "bayer"
    dither_levels: int = 4
    phosphor_amount: float = 0.35
    phosphor_style: str = "rgb"
    phosphor_width: int = 3
    bloom_amount: float = 0.35
    bloom_radius: float = 2.5
    bloom_threshold: float = 0.6
    vignette_amount: float = 0.35
    vignette_power: float = 1.8
    chroma_shift_x: float = 1.0
    chroma_shift_y: float = 0.5
    barrel_amount: float = 0.12
    barrel_falloff: float = 1.0
    noise_amount: float = 0.15
    noise_mode: str = "rgb"
    noise_grain: float = 1.0
    h_jitter_amount: float = 2.0
    h_jitter_speed: float = 2.0
    v_roll_amount: float = 8.0
    v_roll_speed: float = 0.35
    bleed_amount: float = 0.25
    bleed_radius: float = 1.0
    bleed_direction: str = "horizontal"


class RenderCancelled(RuntimeError):
    pass

__all__ = ["RenderSettings", "RenderCancelled"]
