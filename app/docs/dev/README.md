# Oscimorph Developer Docs

This folder contains technical documentation for maintaining and extending Oscimorph.

## Documents

- `docs/dev/architecture.md`: high-level architecture and runtime flow
- `docs/dev/gui.md`: UI structure, preview system, and event wiring
- `docs/dev/render-pipeline.md`: frame generation and export pipeline details
- `docs/dev/audio.md`: audio analysis and modulation model
- `docs/dev/script-api.md`: Script mode contract and safety notes
- `docs/dev/presets.md`: effect preset schema and behavior
- `docs/dev/known-gaps.md`: current technical debt and recommended next refactors

## Code Entry Points

- App entry: `src/oscimorph/app.py`
- GUI package: `src/oscimorph/gui/`
- Render package: `src/oscimorph/render/`
- Audio analysis: `src/oscimorph/audio.py`

