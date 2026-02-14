# Preset System

Oscimorph presets are JSON files created/loaded from the GUI.

## Location

- Default preset directory: `presets/`
- Built-in examples already live there.

## Save and Load Behavior

Handled in `src/oscimorph/gui/legacy.py`:

- Save: `_save_preset()`
- Load: `_load_preset()`
- Serialize: `_collect_effect_preset()`
- Apply: `_apply_effect_preset()`

## Schema (Version 1)

```json
{
  "version": 1,
  "active_effects": ["smoothing", "displace", "glow"],
  "controls": {
    "mod_displace_x_amount": { "amount": 6.0, "band": "band:1" },
    "mod_displace_y_amount": { "amount": 6.0, "band": "band:2" }
  },
  "smoothing_amount": 0.2,
  "glow_enabled": true
}
```

## Rules

- `active_effects` must be a list.
- `controls` must be an object/map.
- Unknown keys are ignored when applying presets.
- Widget values are type-coerced by control type (double/int/combo/check).

## Error Handling

Load failures show warning dialogs for:

- invalid JSON
- non-object root
- schema mismatches
- filesystem errors

