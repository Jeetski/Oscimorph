# Oscimorph Changelog

## 2026-02-14

### Project refactor and structure

- Refactored monolithic modules into package directories:
  - `src/oscimorph/gui/`
  - `src/oscimorph/render/`
- Moved original files to:
  - `src/oscimorph/gui/legacy.py`
  - `src/oscimorph/render/core.py`
- Added package-level module exports:
  - GUI: `main_window.py`, `preview.py`, `workers.py`, `widgets.py`, `__init__.py`
  - Render: `pipeline.py`, `settings.py`, `modulation.py`, `postfx.py`, `text.py`, `__init__.py`
- Updated import paths after move and fixed internal dynamic import for script loading.

### Branding and assets

- Fixed branding asset root path after refactor so icon/logo resolve correctly from `assets/`.
- Added startup splash popup and iterated its design:
  - Switched from native window frame to borderless "fake window" card.
  - Kept close `X` in top-right inside the splash.
  - Applied brand accent color styling to splash text.

### Startup audio

- Added startup sound playback on launch.
- App now looks for startup audio in `assets/`:
  - Prioritizes `startup.mp3`, then common audio extensions.
  - Plays once on startup via `QMediaPlayer` + `QAudioOutput`.

### Documentation

- Added end-user guide:
  - `docs/guide.md`
- Added developer docs set under `docs/dev/`:
  - `README.md`
  - `architecture.md`
  - `gui.md`
  - `render-pipeline.md`
  - `audio.md`
  - `script-api.md`
  - `presets.md`
  - `known-gaps.md`
- Added this changelog:
  - `docs/changelog.md`

### Startup splash: changelog integration

- Added `View changelog` link to startup splash.
- Clicking it opens an in-app changelog reader window and renders `docs/changelog.md`.

### Repository layout cleanup

- Moved project internals into `app/` to keep repository root minimal.
- Root now keeps launchers and project essentials:
  - `run_oscimorph.bat`
  - `run_oscimorph.sh`
  - `README.md`
  - `.gitignore`
- Moved into `app/`:
  - `src/`, `assets/`, `docs/`, `presets/`, `scripts/`, `legal/`
  - runtime directories: `output/`, `debug/`, `temp/`
  - `pyproject.toml`, `requirements.txt`, and backup file(s)
- Updated launchers to run from `app/` and install from `app/requirements.txt`.
- Updated README and user guide paths to the new structure.
