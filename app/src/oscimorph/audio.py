from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class AudioAnalysis:
    audio: np.ndarray  # shape: (channels, samples)
    sr: int
    frame_hop: int
    band_energies: np.ndarray  # shape: (frames, bands)
    duration: float


def load_and_analyze(
    audio_path: str,
    *,
    fps: int,
    sr: int = 44100,
    bands: int = 5,
    n_fft: int = 2048,
) -> AudioAnalysis:
    audio, sr = librosa.load(audio_path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    frame_hop = max(1, int(sr / fps))
    mono = audio.mean(axis=0)

    stft = librosa.stft(mono, n_fft=n_fft, hop_length=frame_hop, window="hann", center=True)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_edges = np.geomspace(20.0, sr / 2.0, bands + 1)
    energies = []
    for i in range(bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        if not np.any(mask):
            energies.append(np.zeros(mag.shape[1]))
            continue
        band = mag[mask].mean(axis=0)
        energies.append(band)

    band_energies = np.stack(energies, axis=1)
    band_energies = band_energies / (band_energies.max(axis=0, keepdims=True) + 1e-6)

    duration = audio.shape[1] / float(sr)

    return AudioAnalysis(
        audio=audio,
        sr=sr,
        frame_hop=frame_hop,
        band_energies=band_energies,
        duration=duration,
    )


def frame_count(duration: float, fps: int) -> int:
    return int(math.ceil(duration * fps))


def band_at_frame(bands: np.ndarray, index: int) -> np.ndarray:
    if index < bands.shape[0]:
        return bands[index]
    return bands[-1]
