# Oscimorph User Guide

Current release status: `alpha v0.1`

This guide is for creators who want to make oscilloscope-style videos without writing code.

## What Oscimorph Does

Oscimorph turns visual outlines into audio-reactive videos:

- Use media, shapes, text, or scripts as the source geometry.
- React motion and styling to music or to an internal oscillator.
- Export MP4 files for social clips, music visuals, and lyric/logo videos.

## Requirements

- Windows installer build: no separate Python required
- Source checkout: Python 3.11+ and `ffmpeg` on PATH

## Launching the App

### Windows

Installed build:

1. Run the Windows installer.
2. Launch Oscimorph from the Start Menu or desktop shortcut.
3. The app opens full-screen.

Source checkout:

1. Double-click `install_dependencies.bat`.
2. Review the dependency check summary.
3. Choose whether to install/update missing items.
4. Double-click `run_oscimorph.bat`.
5. The launcher checks the project virtual environment and `ffmpeg` before startup.
6. The app opens full-screen.

### macOS/Linux

1. Run `bash install_dependencies.sh`.
2. Review the dependency check summary.
3. Choose whether to install/update missing items.
4. Run `bash run_oscimorph.sh`.
5. The launcher checks Python and required packages before startup.
6. The app launches and writes runtime data under the user data directory.

### Manual launch

```powershell
python -m venv app/.venv
app\.venv\Scripts\python -m pip install -r app/requirements.txt
set PYTHONPATH=app/src
app\.venv\Scripts\python -m oscimorph
```

Startup notes:

- The normal app launch shows the startup splash and plays startup audio when available.
- For automation or smoke tests, `OSCIMORPH_SKIP_STARTUP=1` skips that launch presentation.
- Runtime output, logs, temp files, and user presets are written under the user data directory:
  Windows: `%LOCALAPPDATA%\Oscimorph`
  Other platforms: `~/.oscimorph`

## First Render (Fast Path)

1. In **Inputs & Output**:
- Set **Input Mode** to `Shapes`.
- Set **Audio Source** to `Audio File`.
- Choose an audio file.
- Choose output path.
  Default: `%LOCALAPPDATA%\Oscimorph\output\output.mp4` on Windows.

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

The media picker now filters to those supported formats only.

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
- Picker filter: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.aif`, `.aiff`

### Oscillator

- No external audio file required.
- Choose waveform (`Sine`, `Triangle`, `Square`, `Saw`), frequency, depth, and mix.
- Optional **Oscillator Audio Monitor** can play the generated tone.

## Effects Overview

Effects are modular. You can add/remove them from the Effects panel.

Core modulation effects:

- Smoothing
- Displace (X/Y with band targets)
- Thickness Mod
- Glow (toggle + mod + radius + threshold + blend)
- Threshold Mod
- Warp (amount + speed)
- Rotation (with direction)

Additional style/post effects:

- Trail (amount + decay + blend)
- Flicker (amount + band + style + speed + floor)
- Hue Shift
- Scanline (amount + speed + thickness + spacing + dark/light style)
- Decimate
- Jitter (amount + band + axis + random/stepped style)
- Dither:
  Bayer 8x8, ordered 4x4, or diffusion modes with palette level control and optional audio-reactive amount modulation
- Phosphor Mask (amount + RGB/grille/slot style + stripe width)
- Bloom
- Vignette
- Chromatic Aberration
- Barrel Distortion (amount + falloff)
- Noise (amount + RGB/monochrome mode + grain)
- Horizontal Jitter
- Vertical Roll
- Color Bleed (amount + radius + direction)

Band targets available in modulators:

- `Subs`, `Lows`, `Low Mids`, `High Mids`, `Highs`, `All`, `Oscillator`

## Presets

You can save/load effect states as JSON:

- Click **Save Preset** to write a `.json`.
- Click **Load Preset** to restore.
- Built-in presets are seeded into your user preset folder on first run.
- User preset folder:
  Windows: `%LOCALAPPDATA%\Oscimorph\presets`
  Other platforms: `~/.oscimorph/presets`

## Preview vs Final Render

Preview is fast and useful for tuning motion and effect behavior, but it is not fully final-quality.

- Media mode preview uses a simplified proxy behavior and is not source-faithful.
- Final render uses the full pipeline and encoder.
- Always do a short test render before a final export.

## Playback Controls

- Play/Pause, Stop
- Loop with in/out markers
- Timeline scrubbing
- Mute toggle for preview audio

## Output and Logs

- Default output file:
  Windows: `%LOCALAPPDATA%\Oscimorph\output\output.mp4`
  Other platforms: `~/.oscimorph/output/output.mp4`
- Runtime/render log:
  Windows: `%LOCALAPPDATA%\Oscimorph\debug\oscimorph_run.log`
  Other platforms: `~/.oscimorph/debug/oscimorph_run.log`
- Temp workspace:
  Windows: `%LOCALAPPDATA%\Oscimorph\temp`
  Other platforms: `~/.oscimorph/temp`
- User presets:
  Windows: `%LOCALAPPDATA%\Oscimorph\presets`
  Other platforms: `~/.oscimorph/presets`
- Example scripts are bundled from `app/scripts/`

## Troubleshooting

### App launches but render fails

- Confirm `ffmpeg` works in terminal.
- Check `%LOCALAPPDATA%\Oscimorph\debug\oscimorph_run.log` on Windows.
- Try lower resolution/FPS.
- Try a short render in Shapes mode first to isolate media/script-specific issues.

### Launcher says dependencies are missing

- Run `install_dependencies.bat` (Windows) or `install_dependencies.sh` (macOS/Linux).
- Review the summary and confirm install/update.
- Re-run the launcher after install completes.

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
