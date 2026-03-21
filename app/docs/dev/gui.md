# GUI Internals

The concrete GUI is implemented almost entirely in `src/oscimorph/gui/implementation.py`.

## Main Classes

- `MainWindow`: top-level app shell and orchestration layer
- `PreviewCanvas`: custom `QWidget` that draws the lightweight preview image
- `LoopSlider`: timeline scrubber with loop in/out handles
- `RenderWorker`: `QThread` wrapper around `render_video(...)`
- `AudioAnalysisWorker`: `QThread` wrapper around `load_and_analyze(...)`
- `OscillatorAudioDevice`: optional tone monitor for oscillator mode

## Window Structure

### Preview side

- branding/title row
- preview canvas
- transport controls
- timeline and loop range controls
- time readouts
- render progress/status controls

### Control side

- Inputs and Output
- source-mode-specific controls for Media, Shapes, Text, and Script
- Effects panel
- oscillator controls when audio mode is `osc`

### Startup UI

- frameless splash dialog shown at launch unless `OSCIMORPH_SKIP_STARTUP` is set
- branding image, startup audio, and in-app changelog link

## Visibility Rules

`_update_visibility()` is the main section switcher.

- `media` shows media path controls
- `shapes` shows procedural shape controls
- `text` shows text entry/font controls
- `script` shows script path controls
- `audio_mode == "file"` shows audio file controls
- `audio_mode == "osc"` shows oscillator controls and uses oscillator duration for preview/render timing

## Preview Path

Preview is intentionally fast and approximate.

`_update_preview_frame()` does the following:

1. collects the current UI state
2. resolves preview time and audio band values
3. computes modulation values and oscillator contribution
4. builds preview geometry for shape, text, or script modes
5. hands state to `PreviewCanvas.update_state(...)`

`PreviewCanvas` then renders a proxy image by drawing geometry and applying a subset of effect approximations.

Important consequences:

- preview quality is lower than final render quality
- preview uses fixed `preview_fps` timing instead of output FPS
- media mode preview is limited and does not fully reproduce the final edge pipeline

## Rendering and Workers

- `AudioAnalysisWorker` is created when an audio file must be analyzed without blocking the UI
- `RenderWorker` is created when the user starts a final export
- both workers communicate back through Qt signals

## Presets and UI Ownership

Preset logic currently belongs to the GUI layer.

Key methods:

- `_collect_effect_preset()`
- `_apply_effect_preset()`
- `_save_preset()`
- `_load_preset()`

## Coupling Notes

The GUI still imports several underscored helpers from `oscimorph.render`.
That keeps the preview path moving, but it means GUI and render internals are still tightly coupled and should be treated carefully during refactors.
