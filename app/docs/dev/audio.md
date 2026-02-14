# Audio Analysis and Modulation

Implementation: `src/oscimorph/audio.py`.

## Analysis Function

`load_and_analyze(audio_path, fps, sr=44100, bands=5, n_fft=2048) -> AudioAnalysis`

### Steps

1. Load with librosa (`mono=False`).
2. If mono input, duplicate to stereo-like shape `(2, n)`.
3. Compute frame hop from output FPS (`sr / fps`).
4. Run STFT (Hann window).
5. Split frequency bins into geometric band ranges from 20 Hz to Nyquist.
6. Compute mean magnitude per band and normalize each band channel.

## Output Structure

`AudioAnalysis` dataclass:

- `audio`: raw float waveform array, shape `(channels, samples)`
- `sr`: sample rate
- `frame_hop`: hop length used for frame alignment
- `band_energies`: shape `(frames, bands)`, normalized 0..1
- `duration`: seconds

## Frame Access

- `frame_count(duration, fps)` returns output frame count.
- `band_at_frame(bands, index)` clamps to last frame when index exceeds range.

## Band Naming in UI and Scripts

Band indices map to:

- `0`: subs
- `1`: lows
- `2`: low_mids
- `3`: high_mids
- `4`: highs

Plus:

- `all`: mean of all bands
- `osc`: oscillator value (when used)

## Oscillator Path

If `audio_mode == "osc"`, render path generates synthetic band values based on waveform settings instead of audio file analysis.

