# Render Pipeline

Core implementation: `src/oscimorph/render/core.py`

Public entry point: `render_video(settings, progress_cb=None, cancel_cb=None)`

## Pipeline Stages

### 1. Setup

- normalize temp directory handling through `ensure_temp_dir(...)`
- create or rotate the progress log through `init_progress_log(...)`
- resolve source mode from `RenderSettings.media_mode`

### 2. Source preparation

Depending on mode, the pipeline prepares one of the following:

- media frames loaded from image, GIF, or video
- procedural geometry rasterized from shape settings
- text converted to normalized polylines
- script-generated polylines evaluated frame-by-frame

### 3. Audio preparation

- file mode calls `load_and_analyze(...)`
- oscillator mode synthesizes modulation values from waveform, frequency, depth, and mix settings

### 4. Per-frame render loop

For each output frame, the pipeline:

1. computes `t` from frame index and output FPS
2. resolves band energies and oscillator value
3. builds the source frame or polyline geometry
4. applies modulation-driven transforms
5. runs edge extraction and optional overlays
6. applies image-space post effects
7. stores the frame for final export

### 5. Export

- create `ImageSequenceClip`
- attach original audio when `audio_mode == "file"`
- encode through FFmpeg via MoviePy

## Source Modes

### Media

Supported image inputs:

- `.gif`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`

Supported video inputs:

- `.mp4`
- `.mov`
- `.mkv`
- `.avi`
- `.webm`

Unsupported formats raise `RuntimeError("Unsupported media format")`.

### Shapes

The renderer can synthesize geometry for:

- ring
- polygon
- ellipse
- heart
- star
- rectangle
- spiral
- lemniscate
- cardioid
- clover
- superellipse

### Text

Text mode uses Qt outline extraction, then normalizes the resulting paths into `[-1, 1]`-style polyline space.

### Script

Script mode loads a trusted local Python file and calls `generate(t, audio, settings)` for each frame.

## Modulation Semantics

Helpers in `modulation.py` provide:

- band selector resolution
- oscillator value synthesis
- audio/oscillator mixing
- rotation direction policy
- hue-shifted output color

Selectors currently supported by the helper path:

- `all`
- `low`
- `mid`
- `high`
- `band:<index>`
- `osc`

## Post-Processing

The final render path can apply:

- bloom
- vignette
- phosphor mask with selectable RGB, grille, or slot styles and stripe width
- chromatic aberration
- barrel distortion with configurable falloff
- noise with RGB/mono modes and grain control
- horizontal jitter
- vertical roll
- color bleed with radius and direction controls
- dither with selectable Bayer, ordered, or diffusion modes and configurable palette levels

The pipeline also includes earlier modulation-style effects such as thickness, glow, threshold, warp, rotation, trail, flicker, scanline, jitter, and audio-reactive dither modulation. Several of those now expose richer controls in both preview and final render, including glow blend/threshold, trail decay/blend, flicker style/speed/floor, scanline thickness/spacing/style, jitter axis/style, and barrel falloff.

## Progress Tracking

Progress behavior is now implemented in `src/oscimorph/render/progress.py`.

`ProgressTracker` combines three contributors:

- render loop progress
- MoviePy audio chunk progress
- MoviePy frame encode progress

`MoviepyLogger` forwards MoviePy bar updates into that tracker.

Progress is appended to `app/debug/oscimorph_run.log` as percentage lines.

## Cancellation

If `cancel_cb` returns `True`, the pipeline raises `RenderCancelled`.
