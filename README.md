# Chatterbox TTS Studio

A local web app for **Chatterbox** (Resemble AI's open-source TTS): type text → pick a voice → get speech.

**English + हिन्दी**, voice cloning from your own reference clips, expressive controls, long-text handling, and GPU acceleration with automatic CPU fallback. Everything runs 100% locally; generated audio carries Chatterbox's built-in PerTh watermark.

## Quick start

```bat
start.bat
```

Then open **http://127.0.0.1:8488**

First run does three slow things (automatic, one-time):

1. Creates a Python venv is already set up (see manual steps below if missing).
2. Bootstraps 6 voice presets (`scripts/make_presets.py`): downloads one public-domain clip, and for the rest synthesizes the built-in voice and pitch-shifts it into distinct-sounding presets.
3. Downloads the multilingual Chatterbox weights (~2–3 GB) from Hugging Face and loads the engine (CUDA fp16 if the GPU has enough free memory, else CPU).

Subsequent starts load from cache in ~20–60 s.

## Manual setup (if `.venv` doesn't exist)

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

(Use `--index-url https://download.pytorch.org/whl/cpu` on machines without an NVIDIA GPU. On Windows, `pkuseg`/`pykakasi` are intentionally omitted — see requirements.txt.)

## Using the app

- **1 · Text** — type or paste (up to 5,000 chars), with an **Auto / English / हिन्दी** language selector (Auto detects Devanagari script). Long text is split into sentence-sized chunks (≤250 chars) and stitched with 120 ms gaps, so long generations don't destabilize the model.
- **2 · Voice** — click a card to select; ▶ previews the reference clip. **＋ Add a voice** clones a new voice from any uploaded clip (5–15 s of clean, single-speaker speech works best; wav/mp3/m4a/flac/ogg). Custom voices can be deleted (hover the card → ✕).
- **3 · Output** — the audio player appears when generation finishes; download the `.wav` from there.
- **Fine-tuning** —
  - *Exaggeration* (0.25–2): emotion intensity. 0.5 neutral; 0.7+ dramatic. Extreme values can go unstable.
  - *CFG / pace*: lower ≈ slower, more deliberate pacing. Fast-talking references like ~0.3.
  - *Temperature*: sampling randomness. 0.8 default.

The status pill in the header shows engine state (`loading model…` → `ready · CUDA fp16` or `ready · CPU`).

## GPU / CPU behavior

On startup the engine is loaded on CPU, then placed with a **split layout**: the Llama backbone (`t3`, the actual bottleneck) goes to CUDA in fp16 (~550 MB); the vocoder (`s3gen`) and voice encoder stay on CPU fp32. Putting everything on a 2 GB card overflows VRAM, which makes Windows silently spill to system RAM and slows everything down — the split keeps ~900 MB of VRAM free.

The placement is validated with a tiny probe generation. If the GPU has <1.2 GB free, the probe fails, or a chunk later OOMs, that work falls back to CPU fp32 automatically — generations never fail because of VRAM, they just take longer. Reference-clip encodings are cached per voice, so the first use of a voice pays a one-time ~10–15 s cost. Measured on this machine (MX350 2 GB, t3-fp16): a 4 s English sentence ≈ 36 s, a 1.2 s Hindi sentence ≈ 24 s — roughly 3.5× faster than CPU-only.

## Architecture

```
static/index.html   single-page UI (no build step)
server.py           FastAPI: /api/voices, /api/generate, /api/status + static hosting
scripts/            preset bootstrap + audio quality metrics
v3_engine/          official V3 Hindi Space code (vendored)
Chatterbox-Multilingual-hi/   Hindi finetune weights (t3_hi, s3gen_v3)
chatterbox-0.1.2/   vendored English-only source (kept for reference; not used)
data/
  voices/presets/   reference clips + meta.json
  voices/custom/    your cloned voices
  audio/            generated clips
.venv/              Python 3.11, CUDA torch 2.6.0 (cu126)
```

Engines (both load at startup; Hindi routes to the finetune, English to multilingual):

- **multilingual** — `chatterbox-tts==0.1.3` `ChatterboxMultilingualTTS` (23 languages; this app exposes en/hi).
- **v3** — the Hindi finetune `ResembleAI/Chatterbox-Multilingual-hi` (weights in `Chatterbox-Multilingual-hi/`), assembled through the official demo Space code vendored in `v3_engine/`. Noticeably cleaner Hindi than the broad multilingual checkpoint.

Both engines trim the final speech token's audio (~40 ms of noise emitted just before EOS, the same fix Resemble ships in their V3 demo).

## Notes

- Languages exposed in the UI: English and Hindi (the multilingual model supports 23; add more in `server.py` if needed).
- Reference clips are trimmed to 30 s and peak-normalized before use.
- Don't clone voices you don't have rights to; outputs are watermarked.
