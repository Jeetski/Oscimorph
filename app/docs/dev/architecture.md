# Architecture

Oscimorph is a full-screen PySide6 desktop app with a CPU render pipeline built on NumPy, OpenCV, Pillow, librosa, and MoviePy.

## Layers

1. UI and session orchestration
- `src/oscimorph/gui/implementation.py`
- Owns window creation, theming, startup splash, transport, control wiring, preview state, preset actions, and render kickoff.

2. Audio analysis
- `src/oscimorph/audio.py`
- Loads source audio with librosa, computes per-frame band energies, and exposes frame-aligned lookup helpers.

3. Rendering and export
- `src/oscimorph/render/core.py`
- Builds source frames or polylines, applies modulation and post effects, then exports MP4 through MoviePy/FFmpeg.

## Package Layout Today

The package split is real, but most behavior is still concentrated in two files.

### GUI package

- `src/oscimorph/gui/__init__.py`: public GUI exports
- `src/oscimorph/gui/main_window.py`: re-exports `MainWindow`
- `src/oscimorph/gui/preview.py`: re-exports `PreviewCanvas`
- `src/oscimorph/gui/workers.py`: re-exports worker classes
- `src/oscimorph/gui/widgets.py`: re-exports custom widgets
- `src/oscimorph/gui/implementation.py`: actual implementation for all of the above

### Render package

- `src/oscimorph/render/__init__.py`: public render exports
- `src/oscimorph/render/settings.py`: `RenderSettings` and `RenderCancelled`
- `src/oscimorph/render/modulation.py`: band selection, oscillator math, hue shift
- `src/oscimorph/render/postfx.py`: image-space post-processing helpers
- `src/oscimorph/render/text.py`: script loading, text outline extraction, polyline transforms
- `src/oscimorph/render/progress.py`: progress logger and temp directory setup
- `src/oscimorph/render/pipeline.py`: compatibility export for `render_video`
- `src/oscimorph/render/core.py`: main render pipeline implementation

## Runtime Flow

1. `python -m oscimorph` enters `src/oscimorph/__main__.py`.
2. `main()` in `src/oscimorph/app.py` creates `QApplication`.
3. `MainWindow` is instantiated and shown full-screen.
4. Unless `OSCIMORPH_SKIP_STARTUP` is truthy, the startup splash and startup sound are shown on launch.
5. UI changes feed the lightweight preview path on the main thread.
6. Audio file analysis is moved onto `AudioAnalysisWorker` for responsive updates.
7. Final export is moved onto `RenderWorker`, which calls `render_video(...)`.
8. Progress signals update the UI while MoviePy writes the final MP4.

## Data Contracts

### `RenderSettings`

Defined in `src/oscimorph/render/settings.py`.

It includes:

- input/output paths
- source mode selection for media, shapes, text, or script
- audio mode selection for file or oscillator
- output size and FPS
- edge/overlay toggles
- modulation targets and amounts
- oscillator parameters
- post-processing controls

### `AudioAnalysis`

Defined in `src/oscimorph/audio.py`.

Fields:

- `audio`: waveform array shaped `(channels, samples)`
- `sr`: sample rate
- `frame_hop`: analysis hop used for frame alignment
- `band_energies`: normalized band matrix shaped `(frames, bands)`
- `duration`: source duration in seconds

## Threading Model

- Main thread: window, controls, preview state, dialogs, and transport
- `AudioAnalysisWorker`: background analysis for audio-file preview updates
- `RenderWorker`: background final render and export

## Non-Code Runtime Support

- Root launcher scripts perform dependency preflight before startup.
- The render pipeline writes progress lines into `app/debug/oscimorph_run.log`.
- `app/scripts/smoke_startup.py` launches and closes the window quickly for CI smoke coverage.
