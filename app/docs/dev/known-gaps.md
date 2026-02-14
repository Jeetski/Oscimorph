# Known Gaps and Next Refactors

This file tracks practical technical debt in the current codebase.

## 1) Package split is mostly structural

The project now has `gui/` and `render/` packages, but:

- `src/oscimorph/gui/legacy.py` still contains most GUI behavior.
- `src/oscimorph/render/core.py` still contains most render behavior.

Recommendation:

- Extract concrete modules incrementally (no behavior changes), starting with:
  - GUI: startup dialog, preview orchestration, transport controls, preset service
  - Render: input sources, geometry transforms, overlays, postfx chain, progress IO

## 2) Preview and final render parity

Preview is intentionally approximate and not a full parity path:

- Media mode preview is limited.
- Edge extraction/overlay behavior differs from final output.
- Preview runs with fixed `preview_fps` while final render can use user FPS.

Recommendation:

- Add optional "high-fidelity preview" path for sampled frames using render pipeline at low resolution.

## 3) Script execution trust model

Script mode uses `exec()` with limited globals but still trusted local code execution.

Recommendation:

- Document trust assumptions in UI.
- Optional future sandbox mode for untrusted scripts.

## 4) Testing coverage

No automated tests were found for:

- Render output invariants
- Preset compatibility
- Script API contract
- UI state transitions

Recommendation:

- Start with deterministic unit tests on pure helpers in render/audio.
- Add regression snapshot tests for small render fixtures.

## 5) API boundaries

GUI imports several underscored render helpers (private-by-convention functions), creating tight coupling.

Recommendation:

- Introduce explicit public service modules for preview math and shared transforms.
- Reduce direct imports from internal render internals.

