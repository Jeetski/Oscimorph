from __future__ import annotations

import os

from oscimorph.render import RenderSettings, render_video


def main() -> int:
    root_dir = os.getcwd()
    output_dir = os.path.join(root_dir, "app", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "smoke_render.mp4")

    if os.path.exists(output_path):
        os.remove(output_path)

    settings = RenderSettings(
        media_path="",
        audio_path="",
        output_path=output_path,
        media_mode="shapes",
        shape_type="ring",
        audio_mode="osc",
        osc_duration=0.6,
        width=320,
        height=180,
        fps=12,
        enable_line=False,
        enable_lissajous=False,
        preserve_aspect=True,
        glow_strength=1.15,
        glow_radius=3.2,
        glow_threshold=0.12,
        glow_blend="screen",
        trail_strength=0.28,
        trail_decay=0.92,
        trail_blend="lighten",
        flicker_amount=0.35,
        flicker_band="all",
        flicker_style="square",
        flicker_speed=2.2,
        flicker_floor=0.72,
        scanline_amount=0.9,
        scanline_speed=1.4,
        scanline_thickness=2,
        scanline_spacing=4,
        scanline_style="dark",
        jitter_amount=1.5,
        jitter_band="all",
        jitter_axis="y",
        jitter_style="stepped",
        noise_amount=0.18,
        noise_mode="mono",
        noise_grain=2.5,
        bleed_amount=0.35,
        bleed_radius=1.8,
        bleed_direction="both",
        dither_amount=0.65,
        dither_mod_amount=0.25,
        dither_mod_band="all",
        dither_mode="diffusion",
        dither_levels=5,
        phosphor_amount=0.5,
        phosphor_style="slot",
        phosphor_width=2,
        barrel_amount=0.16,
        barrel_falloff=1.4,
    )

    render_video(settings)

    if not os.path.exists(output_path):
        raise RuntimeError("Smoke render did not create an output file")
    if os.path.getsize(output_path) <= 0:
        raise RuntimeError("Smoke render output file is empty")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
