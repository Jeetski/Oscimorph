# Preset System

Oscimorph presets are JSON snapshots of effect-panel state created and loaded from the GUI.

## Location

- built-in presets live in `app/presets/`
- saved presets can be written anywhere the user chooses

## Ownership

Preset logic currently lives in `src/oscimorph/gui/implementation.py`.

Relevant methods:

- `_collect_effect_preset()`
- `_apply_effect_preset()`
- `_save_preset()`
- `_load_preset()`

## What Gets Saved

Presets are focused on effect state, not the full project/session.

They typically capture:

- active effect list
- effect parameter values
- control metadata needed to restore modulated controls

This now includes richer effect controls such as dither mode and palette levels, glow blend/threshold, trail decay/blend, scanline thickness/spacing/style, noise mode/grain, jitter axis/style, bleed radius/direction, flicker style/speed/floor, phosphor mask style/stripe width, and barrel falloff.

They do not serve as a complete project file for inputs, audio sources, or all render settings.

## Schema Shape

The format is intentionally permissive so older presets keep loading when new controls are added.

Example shape:

```json
{
  "version": 1,
  "active_effects": ["smoothing", "displace", "glow"],
  "controls": {
    "mod_displace_x_amount": {
      "amount": 6.0,
      "band": "band:1"
    }
  },
  "smoothing_amount": 0.2,
  "glow_enabled": true
}
```

## Compatibility Rules

- root value must be a JSON object
- `active_effects` must be a list when present
- `controls` must be an object when present
- unknown keys are ignored
- known controls are restored using widget-aware coercion

## Failure Handling

Load failures surface through warning dialogs for:

- invalid JSON
- non-object roots
- schema mismatches
- read/write filesystem errors
