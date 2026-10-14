"""Measure audio quality metrics for one or more wav files.

Usage: .venv/Scripts/python scripts/audio_metrics.py file1.wav [file2.wav ...]

Metrics (computed on the middle 60% of each clip, excluding fades/ends):
  dur       duration in seconds
  rms       speech-band loudness (proxy for level)
  hf_ratio  high-frequency (9-12 kHz) to speech-band (0.3-4 kHz) energy ratio.
            Speech has little energy up there; hiss/noise inflates this number.
  clip_pk   peak absolute amplitude
"""
import sys

import numpy as np
import soundfile as sf


def band_hf_ratio(w, sr):
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    hf = spec[(freqs > 9000) & (freqs < 12000)].sum()
    sp = spec[(freqs > 300) & (freqs < 4000)].sum()
    return hf / max(sp, 1e-12)


def seg_rms(w):
    return float(np.sqrt((w**2).mean()))


def main():
    for path in sys.argv[1:]:
        w, sr = sf.read(path)
        n = len(w)
        mid = w[int(n * 0.2) : int(n * 0.8)]
        tail = w[int(n * 0.95) :]
        body_rms = seg_rms(mid)
        print(
            f"{path}: dur {n / sr:6.2f}s | rms {body_rms:.4f} | "
            f"hf_ratio {band_hf_ratio(mid, sr):.4f} | peak {np.abs(w).max():.3f} | "
            f"tail_rms {seg_rms(tail):.4f}"
        )


if __name__ == "__main__":
    main()
