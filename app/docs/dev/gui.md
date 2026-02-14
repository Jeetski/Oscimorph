# GUI Internals

Primary implementation is `src/oscimorph/gui/legacy.py`.

## Main Components

- `MainWindow`: top-level orchestration.
- `PreviewCanvas`: custom QWidget rendering proxy preview visuals.
- `LoopSlider`: timeline with loop in/out handles.
- `RenderWorker`: QThread wrapper for final rendering.
- `AudioAnalysisWorker`: QThread wrapper for audio analysis.
- `OscillatorAudioDevice`: optional oscillator tone monitor output.

## UI Sections

1. Preview area:
- Logo/title row
- Preview canvas
- Transport controls
- Timeline with loop markers
- Time labels

2. Side panel:
- Inputs & Output
- Shapes
- Text
- Script
- Effects
- Oscillator

3. Startup splash:
- Frameless popup dialog with branding and close `X`.

## Visibility Rules

`_update_visibility()` toggles sections based on mode:

- `media`: enables media picker
- `shapes`: shows shape controls
- `text`: shows text controls
- `script`: shows script controls
- `audio=file`: enables audio path picker
- `audio=osc`: shows oscillator controls and timeline duration sync

## Preview Update Path

`_update_preview_frame()`:

1. Collects current settings.
2. Pulls audio bands for current preview time.
3. Computes modulation signals.
4. Builds shape/text/script polylines.
5. Passes state to `PreviewCanvas.update_state(...)`.

`PreviewCanvas._render_image()` handles:

- Geometry drawing (shape or polylines)
- Some effect approximations (trail, scanline, flicker, etc.)
- Post-processing call chain for several effects

## Important Behavior Notes

- Preview uses a proxy renderer; final render is more complete.
- Preview timing uses `preview_fps` (default 30), independent from output FPS control.
- Media mode preview is intentionally limited and not a full media-edge preview.

## Event Wiring

Most controls connect to `_update_preview_frame()`:

- spinboxes: `valueChanged`
- checkboxes: `toggled`
- comboboxes: `currentIndexChanged`

Transport controls update `preview_time` and optionally sync with `QMediaPlayer`.

## Preset Hooks in GUI

The GUI owns preset save/load:

- `_collect_effect_preset()`
- `_apply_effect_preset()`
- `_save_preset()`
- `_load_preset()`

