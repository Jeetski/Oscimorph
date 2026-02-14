# Render Pipeline

Core implementation: `src/oscimorph/render/core.py`.

Public entry point: `render_video(settings, progress_cb=None, cancel_cb=None)`.

## Pipeline Stages

1. Setup:
- Ensure temp dir and progress log
- Resolve source mode:
  - media frames
  - procedural shape
  - script-generated polylines
  - text outlines

2. Audio source:
- File mode: `load_and_analyze(...)` from `audio.py`
- Osc mode: synthetic band envelope based on oscillator settings

3. Per-frame loop:
- Pick frame/source geometry
- Compute modulation signals from selected bands
- Build edge image
- Optional overlays:
  - waveform line
  - lissajous overlay
- Apply post effects chain
- Store RGB frame for output

4. Export:
- Build `ImageSequenceClip`
- Attach original audio file in file mode
- Encode with `libx264` (`aac` audio when applicable)

## Source Inputs

Media loading supports:

- images: `.gif`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`
- videos: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`

Unsupported formats raise `RuntimeError("Unsupported media format")`.

## Geometry Modes

- Shapes: generated white outlines on black background.
- Script: user function returns normalized polylines.
- Text: Qt text outline extraction to normalized polylines.

## Edge and Overlay

- Edge methods: Sobel or Canny
- Render modes:
  - `edge_only`
  - `edge_overlay`

In `edge_overlay`, waveform and lissajous can be drawn over the edge render.

## Modulation

`_mod_value(...)` resolves a selector and optional oscillator mix:

- `all`
- `band:<index>`
- `osc`

Rotation has direction policy:

- clockwise
- counterclockwise
- alternate

## Post Effects (Final Render)

Available in pipeline:

- bloom
- vignette
- phosphor mask
- color bleed
- chromatic aberration
- barrel distortion
- horizontal jitter
- vertical roll
- noise
- dither
- plus other upstream modulation effects

## Progress Reporting

`_ProgressTracker` combines:

- render frame progress
- moviepy audio chunk progress
- moviepy frame encode progress

Writes percentage lines to `debug/oscimorph_run.log`.

## Cancellation

If `cancel_cb` returns true during render, `RenderCancelled` is raised.

