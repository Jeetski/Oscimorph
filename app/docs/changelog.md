# Oscimorph Changelog

## 2026-03-21

### Effect control expansion

- Expanded `glow`, `scanline`, `noise`, `trail`, `jitter`, `bleed`, `flicker`, `phosphor`, and `barrel` with deeper controls in both preview and final render.
- Added flicker styles with speed/floor control.
- Added phosphor mask styles and stripe width control.
- Added barrel falloff control.
- Updated the Windows smoke render so it exercises the richer post-effect stack.

### Dither upgrade

- Expanded the dither effect with multiple modes:
  - Bayer 8x8
  - Ordered 4x4
  - Diffusion
- Added palette level reduction controls for the dither stage.
- Added audio-reactive dither amount modulation.
- Updated preview and final render paths to use the same dither settings.

### Documentation cleanup

- Refreshed developer docs to describe the current code layout more accurately:
  - `docs/dev/README.md`
  - `docs/dev/architecture.md`
  - `docs/dev/gui.md`
  - `docs/dev/render-pipeline.md`
  - `docs/dev/presets.md`
  - `docs/dev/script-api.md`
  - `docs/dev/known-gaps.md`
- Updated the user guide to better match the installer/launcher flow and current runtime behavior.

### Startup smoke coverage

- Added a lightweight startup smoke script:
  - `app/scripts/smoke_startup.py`
- Added a GitHub Actions workflow for macOS startup verification:
  - `.github/workflows/macos-smoke.yml`
- Smoke flow skips the startup splash/audio using `OSCIMORPH_SKIP_STARTUP=1`.

## 2026-02-14

### Project refactor and structure

- Refactored monolithic modules into package directories:
  - `src/oscimorph/gui/`
  - `src/oscimorph/render/`
- Moved original files to:
  - `src/oscimorph/gui/implementation.py`
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

### Dependency installer split and launcher preflight

- Added dedicated dependency installer scripts at repo root:
  - `install_dependencies.bat`
  - `install_dependencies.sh`
- Installer flow now checks first, then reports full status for:
  - Python 3.11+, pip, ffmpeg
  - each package in `app/requirements.txt` as `missing`, `outdated`, or `satisfied`
- Installer now asks for confirmation before performing any install/update actions.
- Simplified launchers to be launch-only:
  - `run_oscimorph.bat`
  - `run_oscimorph.sh`
- Launchers now run preflight checks and direct users to installer scripts when dependencies are missing.
- Updated docs for the installer-first workflow:
  - `README.md`
  - `app/docs/guide.md`
