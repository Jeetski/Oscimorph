# Oscimorph

Oscimorph turns media, shapes, or scripts into oscilloscope-style, audio-reactive MP4 videos.

Developed by David Cody - Honeycomb Lab
Honeycomb Lab: https://www.honeycomblab.art

## Requirements
- Python 3.11+
- `ffmpeg` on PATH (for reliable MP4 output)

## Install
```powershell
python -m pip install -r requirements.txt
```

## Run
```powershell
python -m oscimorph
```
Or use `run_oscimorph.bat` (auto-installs deps and logs to `debug/`).

## Inputs
Choose one of these:
1. Media: GIF, image, or video
2. Shapes: Ring or Polygon (outline only)
3. Script: Procedural geometry from a `.py` file

Audio input is required for all modes.

## Preview
The preview is lightweight and shows modulation behavior, not final pixels.
- Media mode: ring proxy
- Shapes mode: selected shape
- Script mode: your script geometry

Transport controls include play/pause, stop, loop, in/out, and mute.

## Shapes
- Ring: audio-modulated ring
- Polygon: set sides and orientation
- Orientation modulation is available in Effects

## Effects
- Modulate displacement, thickness, glow, threshold, warp, warp speed, and rotation
- Modulators can target: `subs`, `lows`, `low_mids`, `high_mids`, `highs`, `all`
- Smoothing optionally dampens band motion

## Script input (procedural geometry)
Provide a `.py` file with a `generate` function:

```python
def generate(t, audio, settings):
    # Return a list of polylines, each polyline is a list of (x, y) tuples.
    # Coordinates are normalized in [-1.0, 1.0].
    return [
        [(-1.0, 0.0), (1.0, 0.0)]
    ]
```

`audio` keys:
`subs`, `lows`, `low_mids`, `high_mids`, `highs`, `all`.

`settings` keys:
`width`, `height`, `fps`.

Scripts run as local user code (no sandbox). Keep them fast.

## Output and folders
- Default output: `output/output.mp4`
- Scripts live in `scripts/`
- Logs: `debug/oscimorph_run.log`
- Temp files: `temp/`

## Tips
- If you see a crash on launch, check `debug/oscimorph_run.log`.
- If output is slow, try lowering resolution or FPS.
