# Oscimorph Developer Docs

This folder documents the code as it exists now, not the fully modular shape the project is moving toward.

## Read This First

- The app is launched by `python -m oscimorph` or the root launcher scripts.
- `src/oscimorph/gui/implementation.py` still contains nearly all concrete GUI behavior.
- `src/oscimorph/render/core.py` still contains nearly all concrete render behavior.
- The smaller modules in `src/oscimorph/gui/` and `src/oscimorph/render/` are mostly public import shims and refactor landing zones.

## Documents

- `docs/dev/architecture.md`: runtime flow, package boundaries, and current layering
- `docs/dev/gui.md`: main window structure, preview path, workers, and UI-specific behavior
- `docs/dev/render-pipeline.md`: frame generation, modulation, post-processing, export, and progress tracking
- `docs/dev/audio.md`: librosa-based analysis model and band semantics
- `docs/dev/script-api.md`: Script mode contract, helper environment, and trust model
- `docs/dev/presets.md`: preset storage, schema behavior, and compatibility expectations
- `docs/dev/known-gaps.md`: practical debt and low-risk refactor targets

## Core Entry Points

- App bootstrap: `src/oscimorph/app.py`
- CLI/module entry: `src/oscimorph/__main__.py`
- GUI exports: `src/oscimorph/gui/__init__.py`
- GUI implementation: `src/oscimorph/gui/implementation.py`
- Render exports: `src/oscimorph/render/__init__.py`
- Render implementation: `src/oscimorph/render/core.py`
- Audio analysis: `src/oscimorph/audio.py`

## Repo Support Files

- Root launchers: `run_oscimorph.bat`, `run_oscimorph.sh`
- Dependency installers: `install_dependencies.bat`, `install_dependencies.sh`
- Startup smoke script: `app/scripts/smoke_startup.py`
- GitHub Actions smoke workflow: `.github/workflows/macos-smoke.yml`
