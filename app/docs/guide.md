# Oscimorph User Guide

This guide is for creators who want to make oscilloscope-style videos without writing code.

## What Oscimorph Does

Oscimorph turns visual outlines into audio-reactive videos:

- Use media, shapes, text, or scripts as the source geometry.
- React motion and styling to music or to an internal oscillator.
- Export MP4 files for social clips, music visuals, and lyric/logo videos.

## Requirements

- Python 3.11+
- FFmpeg available on PATH (recommended for reliable MP4 export)

## Launching the App

### Windows

1. Double-click `run_oscimorph.bat`.
2. Wait for dependency install (first launch can take time).
3. App opens full-screen.

### macOS/Linux

1. Run `bash run_oscimorph.sh`.
2. Wait for dependency install.
3. App launches and logs to `app/debug/oscimorph_run.log`.

### Manual launch

```powershell
python -m pip install -r app/requirements.txt
set PYTHONPATH=app/src
python -m oscimorph
```

## First Render (Fast Path)

1. In **Inputs & Output**:
- Set **Input Mode** to `Shapes`.
- Set **Audio Source** to `Audio File`.
- Choose an audio file.
- Choose output path (default is `app/output/output.mp4`).

2. In **Shapes**:
- Pick a shape (`Ring`, `Polygon`, `Star`, etc).
- Adjust size and rotation.

3. In **Effects**:
- Keep defaults first (`smoothing`, `displace`, `thickness`, `glow`, `threshold`, `warp`, `rotation`).
- Add one extra effect at a time from the dropdown.

4. In **Render settings**:
- Set resolution and FPS.
- Choose `Edge Overlay` if you want waveform/lissajous overlays.

5. Click **Render**.

## Input Modes

### Media

- Supported images/animation: `.gif`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`
- Supported videos: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`
- Use when you want edge extraction from existing footage or artwork.

### Shapes

Built-in shapes:

- Ring
- Polygon
- Ellipse
- Heart
- Star
- Rectangle
- Spiral
- Lemniscate
- Cardioid
- Clover
- Superellipse

### Text

- Enter text.
- Pick scale and font.
- Text is converted to outlines and animated like other geometry.

### Script

- Select a `.py` file that provides `generate(t, audio, settings)`.
- Best for procedural visuals and generative experiments.

## Audio Source Modes

### Audio File

- Uses analyzed band energies from your selected audio track.

### Oscillator

- No external audio file required.
- Choose waveform (`Sine`, `Triangle`, `Square`, `Saw`), frequency, depth, and mix.
- Optional oscillator audio monitor can play generated tone.

## Effects Overview

Effects are modular. You can add/remove them from the Effects panel.

Core modulation effects:

- Smoothing
- Displace (X/Y with band targets)
- Thickness Mod
- Glow (toggle + mod + radius)
- Threshold Mod
- Warp (amount + speed)
- Rotation (with direction)

Additional style/post effects:

- Trail
- Flicker
- Hue Shift
- Scanline
- Decimate
- Jitter
- Dither
- Phosphor Mask
- Bloom
- Vignette
- Chromatic Aberration
- Barrel Distortion
- Noise
- Horizontal Jitter
- Vertical Roll
- Color Bleed

Band targets available in modulators:

- `Subs`, `Lows`, `Low Mids`, `High Mids`, `Highs`, `All`, `Oscillator`

## Presets

You can save/load effect states as JSON:

- Click **Save Preset** to write a `.json`.
- Click **Load Preset** to restore.
- Built-in presets are in `app/presets/`.

## Preview vs Final Render

Preview is fast and useful for tuning motion and effect behavior, but it is not fully final-quality.

- Media mode preview is limited (UI notes media cannot be fully previewed).
- Final render uses the full pipeline and encoder.
- Always do a short test render before a final export.

## Playback Controls

- Play/Pause, Stop
- Loop with in/out markers
- Timeline scrubbing
- Mute toggle for preview audio

## Output and Logs

- Default output file: `app/output/output.mp4`
- Runtime/render log: `app/debug/oscimorph_run.log`
- Temp workspace: `app/temp/`

## Troubleshooting

### App launches but render fails

- Confirm `ffmpeg` works in terminal.
- Check `app/debug/oscimorph_run.log`.
- Try lower resolution/FPS.

### No reaction to audio

- Confirm audio file is selected in `Audio File` mode.
- Check modulator bands are set to `All` or a non-zero band.
- Increase effect amounts (for example displace/thickness/warp).

### Script mode errors

- Ensure script exists and defines `generate(t, audio, settings)`.
- Check error popup and log.

### Text mode issues

- Use a font installed on your system.
- Keep text non-empty.

## Practical Workflow Tips

- Add effects one by one and test quickly.
- Save versions of presets often.
- Start low-res (for example 1280x720, 30 FPS), then final render at target resolution.
- For social edits, render short clips first to lock look and timing.
