# Script Mode API

Script mode lets users supply a Python file that generates geometry each frame.

Implementation entry: `_load_script(...)` and script execution in `src/oscimorph/render/core.py`.

## Required Function

Your script must define:

```python
def generate(t, audio, settings):
    return [
        [(-1.0, 0.0), (1.0, 0.0)]
    ]
```

## Parameters

- `t`: current time in seconds (float)
- `audio`: dict of normalized audio/oscillator values
- `settings`: render settings subset

### `audio` keys

- `subs`
- `lows`
- `low_mids`
- `high_mids`
- `highs`
- `all`
- `osc`

### `settings` keys

- `width`
- `height`
- `fps`

## Return Format

- Return a list of polylines.
- Each polyline is a list of `(x, y)` tuples.
- Coordinates are normalized (typically in `[-1, 1]` range).

## Error Cases

- Missing file: `Script file not found`
- Missing callable: `Script must define a callable generate(t, audio, settings)`
- Runtime exceptions bubble up and are shown in GUI warning dialogs.

## Security Note

Scripts are executed with `exec(...)`. This is trusted local execution and should only be used with scripts from trusted sources.

