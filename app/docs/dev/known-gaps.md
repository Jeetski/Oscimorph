# Known Gaps and Next Refactors

This document tracks the highest-value cleanup targets that can be addressed without redefining the product.

## 1. The package split is still mostly structural

Current state:

- `src/oscimorph/gui/implementation.py` remains the real GUI module
- `src/oscimorph/render/core.py` remains the real render module
- most neighboring files are export shims or helper extractions

Recommended next steps:

- split GUI by responsibility, not widget type
- split render by pipeline stage, not just helper category
- keep imports stable through package re-exports while moving code

Good low-risk extraction targets:

- GUI startup splash and changelog window
- preview orchestration and transport
- preset service
- render source loaders
- polyline transforms
- overlay drawing
- post-processing chain assembly

## 2. Preview and final render still differ in meaningful ways

Current state:

- preview is intentionally approximate
- media mode preview is limited
- preview timing is driven by `preview_fps`, not output FPS
- some effects are approximated differently from the final pipeline

Recommended next step:

- add an optional sampled low-resolution render-preview path for frame-accurate checks

## 3. GUI and render internals are tightly coupled

Current state:

- GUI imports several underscored helpers from `oscimorph.render`
- preview math depends on render internals rather than a stable public service layer

Recommended next steps:

- introduce an explicit preview/render shared service module
- stop treating private helper imports as long-term API

## 4. Script mode is trusted execution only

Current state:

- scripts are loaded with `exec(...)`
- helper globals are limited, but not sandboxed

Recommended next steps:

- keep the trust model documented in the UI and docs
- consider a future restricted mode only if untrusted scripts become a real use case

## 5. Test coverage is still thin

There is now a startup smoke path for macOS CI, but broader automated coverage is still missing.

High-value missing coverage:

- render output invariants
- preset round-trip compatibility
- script API contract checks
- UI visibility/state transitions
- regression tests for representative source modes

Recommended next steps:

- add unit tests for pure helpers in `audio.py`, `modulation.py`, `postfx.py`, and `text.py`
- add tiny render fixtures for deterministic snapshot-style checks
- expand smoke coverage beyond startup success
