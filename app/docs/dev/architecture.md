# Architecture

Oscimorph is a desktop PySide6 application with a CPU-based render pipeline.

## Layers

1. UI + interaction:
- `src/oscimorph/gui/`
- Owns controls, preview state, playback controls, and render job kickoff.

2. Audio analysis:
- `src/oscimorph/audio.py`
- Loads audio, computes normalized multi-band energies, and provides frame-index lookup.

3. Render pipeline:
- `src/oscimorph/render/`
- Builds source frames, applies edge + overlays + post effects, and writes MP4.

## Current Package Layout

The code was recently split into packages:

- `src/oscimorph/gui/__init__.py`: public GUI exports.
- `src/oscimorph/gui/main_window.py`, `preview.py`, `workers.py`, `widgets.py`: compatibility exports.
- `src/oscimorph/gui/legacy.py`: still contains almost all concrete GUI logic.

- `src/oscimorph/render/__init__.py`: public render exports.
- `src/oscimorph/render/pipeline.py`, `settings.py`, `modulation.py`, `postfx.py`, `text.py`: compatibility exports.
- `src/oscimorph/render/core.py`: still contains almost all concrete rendering logic.

## Runtime Flow

1. `python -m oscimorph` calls `main()` in `src/oscimorph/app.py`.
2. `QApplication` is created.
3. `MainWindow` is created, themed, and shown full-screen.
4. User changes controls; preview updates from a lightweight preview path.
5. On render:
- GUI builds `RenderSettings`.
- `RenderWorker` thread calls `render_video(settings, ...)`.
- Progress updates flow back via Qt signals.
6. Final MP4 is written via MoviePy + FFmpeg.

## Data Contracts

### RenderSettings

Defined in `src/oscimorph/render/core.py` (re-exported via `src/oscimorph/render/settings.py`).
Contains:

- I/O paths
- Media mode settings
- Audio mode settings
- Resolution/FPS/aspect
- Modulation and post-effect controls

### AudioAnalysis

Defined in `src/oscimorph/audio.py`:

- `audio`: `(channels, samples)`
- `sr`
- `frame_hop`
- `band_energies`: `(frames, bands)`
- `duration`

## Threading Model

- Main UI thread: interaction, preview drawing, controls.
- Render thread (`RenderWorker`): final render execution and callbacks.
- Audio analysis thread (`AudioAnalysisWorker`): analysis for responsive UI preview.

