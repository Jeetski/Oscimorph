# Oscimorph

Oscimorph is a full-screen desktop app for turning media, shapes, text, or scripts into oscilloscope-style, audio-reactive MP4 videos.
Oscimorph treats sound as signal and visuals as geometry — not presets.

Developed by David Cody - Honeycomb Lab  
Honeycomb Lab: https://www.honeycomblab.art

**How It Works**
1. Choose an input mode: Media, Shapes, Text, or Script.
1. Choose an audio source: Audio File or Oscillator.
1. Adjust render settings, overlays, and effects.
1. Render to an MP4.

**Features**
- Input modes: media (GIF, image, or video), shapes, text, or Python script geometry
- Audio source: audio file analysis or internal oscillator (with optional audio monitor)
- Render modes: edge-only or edge-overlay with waveform line and lissajous overlay
- Built-in shapes: ring, polygon, ellipse, heart, star, rectangle, spiral, lemniscate, cardioid, clover, superellipse
- Audio modulation targets: `all`, `low`, `mid`, `high`, `band:<index>`, or `osc`
- Effect stack: smoothing, displacement, thickness, glow, threshold, warp, rotation, trail, flicker, hue shift, scanline, decimate, jitter, dither, phosphor, bloom, vignette, chromatic aberration, barrel distortion, noise, horizontal jitter, vertical roll, color bleed
- Preset save/load (JSON) with built-in presets in `presets/`
- MP4 output via MoviePy/FFmpeg with resolution, FPS, and aspect controls

**Requirements**
- Python 3.11+
- `ffmpeg` on PATH (for reliable MP4 output)

**Quick Start (Windows)**
1. Double-click `run_oscimorph.bat`.
1. The script installs dependencies, launches the app, and logs to `debug/oscimorph_run.log`.

**Quick Start (macOS/Linux)**
1. `bash run_oscimorph.sh`
1. The script installs dependencies, launches the app, and logs to `debug/oscimorph_run.log`.

**Manual Run**
```powershell
python -m pip install -r requirements.txt
set PYTHONPATH=src
python -m oscimorph
```

If you install the package in editable mode, you can also run `oscimorph` directly.

**Input Modes**
- **Media**: Uses a GIF/image/video as the base frame source.
- **Shapes**: Generates procedural outlines (see shapes list above).
- **Text**: Converts a font outline into polylines.
- **Script**: Runs a local Python file that returns polylines per frame.

**Audio Analysis**
Oscimorph analyzes the audio into 5 bands and exposes these to modulators and scripts:
- `subs`, `lows`, `low_mids`, `high_mids`, `highs`, plus `all` (average)
- `osc` is available when mixing the internal oscillator

**Script API**
Provide a `.py` file with a `generate` function:

```python
def generate(t, audio, settings):
    # Return a list of polylines, each polyline is a list of (x, y) tuples.
    # Coordinates are normalized in [-1.0, 1.0].
    return [
        [(-1.0, 0.0), (1.0, 0.0)]
    ]
```

`audio` keys: `subs`, `lows`, `low_mids`, `high_mids`, `highs`, `all`, `osc`  
`settings` keys: `width`, `height`, `fps`

This allows fully procedural, generative, and fractal visuals driven directly by sound.

Example scripts live in `scripts/` (e.g., `lissajous.py`, `spirograph.py`, `julia_set.py`).

**Presets**
- Use the **Effects** panel to save and load JSON presets.
- Built-in presets are stored in `presets/`.

**Output & Logs**
- Default output: `output/output.mp4`
- Render log: `debug/oscimorph_run.log`
- Temporary files: `temp/`

**Controls & Tips**
- Press `Esc` to exit (with confirmation).
- Preview is lightweight and intended to show modulation behavior, not final pixel quality.
- If rendering is slow, reduce resolution or FPS.

**Project Layout**
- `src/oscimorph/`: application code (GUI, audio analysis, renderer)
- `presets/`: effect presets
- `scripts/`: example procedural geometry scripts
- `assets/`: icons and branding
- `output/`, `debug/`, `temp/`: runtime directories

**Who Oscimorph Is For**
- Music producers who want custom visuals for their tracks
- Artists making lyric videos or logo reveals
- Developers and math/graphics nerds who enjoy procedural geometry

Not intended as a VJ performance tool or real-time live visuals (yet).

**License**
MIT License. See `legal/LICENSE.txt`.
