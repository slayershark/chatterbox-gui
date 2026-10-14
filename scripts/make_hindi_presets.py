"""Build additional Hindi voice presets from public-domain sources.

Sources:
  1. Resemble's official second Hindi prompt (chatterbox-demo-samples bucket).
  2. Archive.org Hindi audiobooks (ArvindGupta collection, public domain).

For each candidate we pick the cleanest 8-12 s segment (highest RMS, no
clipping, inside speech intervals), estimate the speaker's median F0 to name
and label the voice, and only keep clips passing the noise gates.

Run:  .venv/Scripts/python scripts/make_hindi_presets.py
Safe to re-run: existing preset files are kept unless --force.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "data" / "voices" / "presets"
S3GEN_SR = 24000

OFFICIAL = [
    ("kunal", "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi/hi_prompts1.flac",
     "Male, Hindi (official Resemble reference)"),
]

# Public-domain Hindi audiobook items on archive.org (ArvindGupta collection).
# (female_name, male_name): the voice is named by its measured median F0.
ARCHIVE_ITEMS = {
    "rutherford-finds-21-b-hindi-audiobook": ("ananya", "kabir"),
    "united-nations-hindi-audiobook-part-i": ("diya", "arjun"),
    "surajmukhi-khilte-hain-jahan": ("tara", "devraj"),
    "someday-a-tree-hindi-audiobook": ("rima", "vidur"),
    "ruby-prince-hindi-audiobook": ("isha", "manav"),
    "vigyan-ke-prayog-keith-warren-part-ii": ("sarika", "vikram"),
}

HF_RATIO_GATE = 0.002  # 9-12 kHz vs speech-band energy; human speech ~0.001
CLIP_GATE = 0.995      # reject segments that touch full scale


def hf_ratio(w: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    hf = spec[(freqs > 9000) & (freqs < 12000)].sum()
    sp = spec[(freqs > 300) & (freqs < 4000)].sum()
    return float(hf / max(sp, 1e-12))


def median_f0(w: np.ndarray, sr: int) -> float:
    f0, _vflag, _vparam = librosa.pyin(y=w, fmin=60, fmax=400, sr=sr, frame_length=2048)
    vals = f0[~np.isnan(f0)]
    return float(np.median(vals)) if len(vals) else 0.0


def best_segment(w: np.ndarray, sr: int, want_s: float = 10.0, min_speech: float = 0.70) -> np.ndarray | None:
    """Pick the highest-RMS clean ~want_s window. Natural narration pauses
    every few seconds, so the window slides across the whole file (1 s steps,
    anchored near speech onsets) and merely needs decent speech coverage."""
    want = int(want_s * sr)
    if len(w) < want + int(0.5 * sr):
        return None
    intervals = librosa.effects.split(y=w, top_db=35)
    if len(intervals) == 0:
        return None
    onsets = [int(s) for s, _e in intervals]
    tol = int(0.5 * sr)

    def speech_covered(seg: np.ndarray) -> int:
        sub = librosa.effects.split(y=seg, top_db=40)
        return sum(int(e) - int(s) for s, e in sub)

    best: np.ndarray | None = None
    best_rms = 0.0
    for pos in range(0, len(w) - want, sr):
        if not any(abs(pos - st) < tol for st in onsets):
            continue
        seg = w[pos : pos + want]
        rms = float(np.sqrt((seg**2).mean()))
        if rms <= best_rms or float(np.abs(seg).max()) >= CLIP_GATE:
            continue
        if speech_covered(seg) > min_speech * want:
            best_rms, best = rms, seg
    return best


def fetch_item_audio(identifier: str) -> tuple[str, bytes] | None:
    """First audio file of the archive.org item (flac/wav/mp3/m4a)."""
    url = f"https://archive.org/metadata/{identifier}"
    with urllib.request.urlopen(url, timeout=30) as r:
        meta = json.load(r)
    for f in meta.get("files", []):
        name = f.get("name", "")
        if name.lower().endswith((".flac", ".wav", ".mp3", ".m4a")):
            dl = f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"
            with urllib.request.urlopen(dl, timeout=180) as r:
                return name, r.read()
    return None


def decode_any(raw: bytes, source: str) -> tuple[np.ndarray, int] | None:
    """Decode flac/wav/mp3 via soundfile, m4a via ffmpeg if present."""
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return audio.mean(axis=1), sr
    except Exception:
        pass
    if source.lower().endswith(".m4a"):
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            return None
        proc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(S3GEN_SR), "pipe:1"],
            input=raw, capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            audio = np.frombuffer(proc.stdout, dtype=np.float32)
            return audio, S3GEN_SR
    return None


def process(raw: bytes, source: str, want_s: float = 10.0) -> tuple[bool, str, str, np.ndarray | None]:
    """Evaluate raw audio. Returns (accepted, name, description, segment)."""
    decoded = decode_any(raw, source)
    if decoded is None:
        return False, "", "undecodable audio", None
    audio, sr = decoded
    if sr != S3GEN_SR:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=S3GEN_SR)
        sr = S3GEN_SR
    seg = best_segment(audio, sr, want_s=want_s)
    if seg is None:
        return False, "", "no clean 10 s segment", None
    peak = float(np.abs(seg).max())
    ratio = hf_ratio(seg, sr)
    f0 = median_f0(seg, sr)
    print(f"    candidate: peak {peak:.2f} | hf {ratio:.4f} | f0 {f0:.0f} Hz | src {source}")
    if peak >= CLIP_GATE or ratio > HF_RATIO_GATE or f0 < 60:
        return False, "", "failed noise/clipping gate", None
    seg = (seg / peak * 0.9).astype(np.float32)
    return True, "", "", seg


def main() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    meta_path = PRESETS / "meta.json"
    meta: dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    print("Building additional Hindi presets...")

    for label, url, desc in OFFICIAL:
        dest = PRESETS / f"{label}.wav"
        if dest.exists() and not force:
            continue
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            ok, _name, why, seg = process(raw, "official", want_s=8.0)
            if ok and seg is not None:
                sf.write(dest, seg, S3GEN_SR, subtype="PCM_16")
                meta[label] = {"description": desc}
                print(f"  preset '{label}' written ({desc})")
            else:
                print(f"  official '{label}' rejected: {why}")
        except Exception as exc:
            print(f"  official '{label}' failed: {exc}")

    made = len([p for p in ARCHIVE_ITEMS.values() if (PRESETS / f"{p[0]}.wav").exists()
                or (PRESETS / f"{p[1]}.wav").exists()])
    wanted = 3
    for ident, (f_name, m_name) in ARCHIVE_ITEMS.items():
        if made >= wanted:
            break
        if (PRESETS / f"{f_name}.wav").exists() or (PRESETS / f"{m_name}.wav").exists():
            continue
        print(f"  item: {ident}")
        try:
            hit = fetch_item_audio(ident)
            if not hit:
                print("    no audio file in item")
                continue
            fname, raw = hit
            ok, _n, why, seg = process(raw, fname)
            if not ok or seg is None:
                print(f"    rejected: {why}")
                continue
            f0 = median_f0(seg, S3GEN_SR)
            gender = "Female" if f0 > 155 else "Male"
            label = f_name if gender == "Female" else m_name
            sf.write(PRESETS / f"{label}.wav", seg, S3GEN_SR, subtype="PCM_16")
            desc = f"{gender}, Hindi (public-domain audiobook)"
            meta[label] = {"description": desc}
            print(f"  preset '{label}' written ({desc})")
            made += 1
        except Exception as exc:
            print(f"    item failed: {exc}")

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done: meta.json now has {len(meta)} entries")


if __name__ == "__main__":
    main()
