"""Screen candidate reference clips and write accepted ones as presets.

Gates: hf_ratio <= 0.002, peak < 0.995, median F0 >= 60 Hz.
Usage: .venv/Scripts/python scripts/screen_voice_candidates.py NAME=FILE NAME=FILE ...
Writes data/voices/presets/<name>.wav + updates meta.json with measured gender.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "data" / "voices" / "presets"
S3GEN_SR = 24000
HF_GATE = 0.002
CLIP_GATE = 0.995


def decode(path: Path) -> tuple[np.ndarray, int]:
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=True)
        return w.mean(axis=1), sr
    except Exception:
        proc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(S3GEN_SR), "pipe:1"],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise
        return np.frombuffer(proc.stdout, dtype=np.float32), S3GEN_SR


def hf_ratio(w, sr):
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    hf = spec[(freqs > 9000) & (freqs < 12000)].sum()
    sp = spec[(freqs > 300) & (freqs < 4000)].sum()
    return hf / max(sp, 1e-12)


def best_segment(w, sr, want_s=10.0, min_speech=0.70):
    want = int(want_s * sr)
    if len(w) < want + int(0.5 * sr):
        return None
    intervals = librosa.effects.split(y=w, top_db=35)
    if len(intervals) == 0:
        return None
    onsets = [int(s) for s, _ in intervals]
    tol = int(0.5 * sr)

    def covered(seg):
        return sum(int(e) - int(s) for s, e in librosa.effects.split(y=seg, top_db=40))

    best, best_rms = None, 0.0
    for pos in range(0, len(w) - want, sr):
        if not any(abs(pos - st) < tol for st in onsets):
            continue
        seg = w[pos : pos + want]
        rms = float(np.sqrt((seg**2).mean()))
        if rms <= best_rms or float(np.abs(seg).max()) >= CLIP_GATE:
            continue
        if covered(seg) > min_speech * want:
            best_rms, best = rms, seg
    return best


def median_f0(w, sr):
    f0, _, _ = librosa.pyin(y=w, fmin=60, fmax=400, sr=sr, frame_length=2048)
    v = f0[~np.isnan(f0)]
    return float(np.median(v)) if len(v) else 0.0


def main() -> None:
    meta_path = PRESETS / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        path = Path(path)
        try:
            w, sr = decode(path)
            if sr != S3GEN_SR:
                w = librosa.resample(y=w, orig_sr=sr, target_sr=S3GEN_SR)
                sr = S3GEN_SR
            seg = best_segment(w, sr)
            if seg is None:
                print(f"[{name}] no clean segment")
                continue
            peak = float(np.abs(seg).max())
            ratio = hf_ratio(seg, sr)
            f0 = median_f0(seg, sr)
            gender = "Female" if f0 > 155 else "Male"
            ok = peak < CLIP_GATE and ratio <= HF_GATE and f0 >= 60
            print(f"[{name}] peak {peak:.2f} | hf {ratio:.4f} | f0 {f0:.0f} Hz -> {gender}"
                  f" {'ACCEPT' if ok else 'REJECT'}")
            if not ok:
                continue
            seg = (seg / peak * 0.9).astype(np.float32)
            sf.write(PRESETS / f"{name}.wav", seg, S3GEN_SR, subtype="PCM_16")
            meta[name] = {"description": f"{gender}, English (LibriVox narrator)"}
        except Exception as exc:
            print(f"[{name}] failed: {exc}")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
