# Script Mode API

Script mode lets a local Python file generate polyline geometry for each frame.

Implementation helpers live in `src/oscimorph/render/text.py`, and the render loop calls the loaded function from `src/oscimorph/render/core.py`.

## Required Function

The script must define a callable named `generate`:

```python
def generate(t, audio, settings):
    return [
        [(-1.0, 0.0), (1.0, 0.0)]
    ]
```

## Function Parameters

- `t`: current time in seconds
- `audio`: normalized audio/oscillator payload
- `settings`: small settings dict with render dimensions and FPS

### Audio payload keys

- `subs`
- `lows`
- `low_mids`
- `high_mids`
- `highs`
- `all`
- `osc`

### Settings payload keys

- `width`
- `height`
- `fps`

## Return Value

Return a list of polylines.

Each polyline is:

- an ordered list of `(x, y)` tuples
- usually expressed in normalized coordinates around `[-1.0, 1.0]`

Closed loops should repeat the start point if the script wants that behavior explicitly.

## Script Environment

The loader currently exposes a lightweight execution environment with:

- normal Python builtins
- `math`
- `random`
- `np` (`numpy`)

No sandbox is applied.

## Error Cases

- missing file: `Script file not found`
- missing callable: `Script must define a callable generate(t, audio, settings)`
- runtime exceptions bubble up to the UI and render flow

## Trust Model

Scripts are executed with `exec(...)`.
Treat Script mode as trusted local code execution only.
