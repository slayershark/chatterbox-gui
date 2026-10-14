"""
Bootstrap voice presets for the TTS Studio.

Strategy:
  1. Try to download small public-domain reference clips (LibriVox-style speech).
  2. Whatever fails (or for guaranteed variety), synthesize a short line with the
     model's built-in voice and pitch/time-shift it into distinct-sounding presets.

Run once:  .venv/Scripts/python scripts/make_presets.py
Safe to re-run: existing preset files are kept unless you pass --force.
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chatterbox-0.1.2" / "src"))

import numpy as np
import soundfile as sf

S3GEN_SR = 24000  # only used for uploads via the UI; presets use model.sr
PRESETS = ROOT / "data" / "voices" / "presets"

META = {
    "nova":  ("Female, warm and clear", 1.00, 1.00),
    "aria":  ("Female, bright and quick", 1.15, 0.94),
    "ember": ("Female, low and calm", 0.80, 1.06),
    "atlas": ("Male, deep and steady", 0.75, 1.02),
    "raven": ("Male, dark and slow", 0.62, 1.10),
    "sage":  ("Male, mid-range neutral", 0.90, 0.98),
}

DEFAULT_TEXT = (
    "Now let's make my mum's favourite. So three mars bars into the pan. "
    "Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. "
    "A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
)


def shift_audio(audio: np.ndarray, sr: int, pitch: float, speed: float) -> np.ndarray:
    """Pitch-shift then time-stretch using phase-vocoder librosa effects."""
    import librosa

    out = audio
    if abs(pitch - 1.0) > 1e-3:
        n_steps = 12 * float(np.log2(pitch))
        out = librosa.effects.pitch_shift(y=out, sr=sr, n_steps=n_steps)
    if abs(speed - 1.0) > 1e-3:
        out = librosa.effects.time_stretch(y=out, rate=speed)
    return out


def trim_silence(audio: np.ndarray, sr: int, margin: float = 0.15) -> np.ndarray:
    import librosa

    intervals = librosa.effects.split(y=audio, top_db=35)
    if len(intervals) == 0:
        return audio
    start, end = intervals[0][0], intervals[-1][1]
    pad = int(margin * sr)
    start = max(0, start - pad)
    end = min(len(audio), end + pad)
    return audio[start:end]


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.95).astype(np.float32)


def try_download(destinations: dict[str, Path]) -> bool:
    """Grab one stable public-domain male speech clip (JFK, 11s, HF CDN)."""
    sources = {
        "atlas": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
    }
    ok = False
    for name, url in sources.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            dest = destinations[name]
            sf.write(dest, audio, sr, subtype="PCM_16")
            print(f"  downloaded preset '{name}' ({len(audio) / sr:.1f}s @ {sr} Hz)")
            ok = True
        except Exception as exc:
            print(f"  download for '{name}' failed ({exc}); will synthesize instead")
    return ok


def main() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    print("Bootstrapping voice presets...")

    # -- 1. optional downloads --------------------------------------------
    dest = {n: PRESETS / f"{n}.wav" for n in META}
    try_download(dest)

    # -- 2. guaranteed offline fallbacks (only for presets still missing) --
    missing = [n for n in META if not (PRESETS / f"{n}.wav").exists()]
    if "--force" in sys.argv or missing:
        from chatterbox.models.s3gen import S3GEN_SR as MODEL_SR
        from chatterbox.tts import ChatterboxTTS

        print("loading model for fallback preset synthesis...")
        t0 = time.time()
        model = ChatterboxTTS.from_pretrained("cpu")
        print(f"model ready in {time.time() - t0:.1f}s")

        cache = PRESETS / "_base.npy"
        if "--force" not in sys.argv and cache.exists():
            base = np.load(cache)
            sr = MODEL_SR
            print(f"reusing cached base clip ({len(base) / sr:.1f}s)")
        else:
            print("synthesizing base clip (this takes a few minutes on CPU)...")
            wav = model.generate(DEFAULT_TEXT, exaggeration=0.5, cfg_weight=0.5)
            base = wav.squeeze(0).cpu().numpy().astype(np.float32)
            sr = model.sr
            base = trim_silence(base, sr)
            base = normalize(base)
            np.save(cache, base)
            print(f"base clip: {len(base) / sr:.1f}s (cached)")

        for name, (desc, pitch, speed) in META.items():
            out = PRESETS / f"{name}.wav"
            if out.exists() and "--force" not in sys.argv:
                continue
            shifted = shift_audio(base, sr, pitch, speed)
            shifted = normalize(shifted)
            sf.write(out, shifted, sr, subtype="PCM_16")
            print(f"  preset '{name}' ({desc}) written")
    else:
        print(f"all {len(META)} presets already exist; pass --force to regenerate.")

    meta_path = PRESETS / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            "{\n" + ",\n".join(f'  "{n}": {{"description": "{d[0]}"}}' for n, d in META.items()) + "\n}\n",
            encoding="utf-8",
        )
        print("meta.json written")


if __name__ == "__main__":
    main()
