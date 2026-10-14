"""
Chatterbox TTS Studio - backend.

Endpoints:
  GET    /api/voices          list voices (presets + user clones)
  POST   /api/voices          create a voice from an uploaded reference clip
  DELETE /api/voices/{name}   remove a user voice
  GET    /api/status          device + engine load state
  POST   /api/generate        synthesize text with the selected voice -> wav file
  GET    /audio/...           generated clips + voice previews
  /                           serve the UI (static/index.html)

Engines:
  "multilingual"  pip chatterbox-tts 0.1.3 ChatterboxMultilingualTTS (en+hi)
  "v3"            Hindi finetune (ResembleAI/Chatterbox-Multilingual-hi) via the
                  official Space code vendored in chatterbox_v3/ (t3_hi + s3gen_v3)

Hindi routes to the finetune when it is ready; English prefers multilingual.
Both engines run split-placement: t3 backbone fp16 on CUDA, s3gen/ve fp32 on
CPU (a 2 GB card cannot hold more). Only one t3 sits on the GPU at a time.
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

AUDIO_SR = 24000  # S3GEN_SR for both engines
S3_TOKEN_RATE = 25  # speech tokens per second

# ---------------------------------------------------------------- config

DATA_DIR = ROOT / "data"
VOICES_DIR = DATA_DIR / "voices"
AUDIO_DIR = DATA_DIR / "audio"
SESSIONS_DIR = AUDIO_DIR / "sessions"  # one subfolder per generation session
PRESETS_DIR = VOICES_DIR / "presets"
USER_VOICES_DIR = VOICES_DIR / "custom"
for d in (PRESETS_DIR, USER_VOICES_DIR, SESSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CUDA_AVAILABLE = torch.cuda.is_available()
GPU_FREE_GB = 0.0
GPU_OK = False
if CUDA_AVAILABLE:
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        GPU_FREE_GB = free_b / 1024**3
        GPU_OK = GPU_FREE_GB > 1.2
    except Exception:
        GPU_OK = True

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML_HANDLE = None
MAX_TEXT_CHARS = 5000
CHUNK_LIMIT = 250
MAX_UPLOAD_SECONDS = 30
SEED = 1706

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def log(msg: str) -> None:
    print(f"[server] {msg}", flush=True)


def set_seed() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


# ---------------------------------------------------------- text helpers

def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def punc_norm_ml(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return text
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = text.replace("।.", "।").replace(".।", "।")
    return text


def split_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > limit:
            head = s[:limit].rsplit(" ", 1)[0] if " " in s[:limit] else s[:limit]
            chunks.append(head)
            s = s[len(head):].strip()
        if len(cur) + len(s) + 1 <= limit:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- voices

class Voice:
    def __init__(self, name: str, wav_path: Path, kind: str, description: str = ""):
        self.name = name
        self.wav_path = wav_path
        self.kind = kind  # "preset" | "custom"
        self.description = description

    def to_dict(self) -> dict:
        dir_name = "presets" if self.kind == "preset" else "custom"
        return {
            "id": self.name,
            "name": self.name.replace("_", " ").title(),
            "kind": self.kind,
            "description": self.description,
            "preview": f"/audio/voices/{dir_name}/{self.wav_path.name}",
        }


def _safe_name(raw: str) -> str:
    name = re.sub(r"[^a-z0-9_-]+", "_", raw.strip().lower()).strip("_")
    return name[:40] or "voice"


def list_voice_dirs() -> list[Voice]:
    voices: list[Voice] = []
    meta_file = PRESETS_DIR / "meta.json"
    meta: dict = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    for wav in sorted(PRESETS_DIR.glob("*.wav")):
        voices.append(Voice(wav.stem, wav, "preset", meta.get(wav.stem, {}).get("description", "")))
    for wav in sorted(USER_VOICES_DIR.glob("*.wav")):
        voices.append(Voice(wav.stem, wav, "custom", "Cloned from your uploaded clip"))
    return voices


def get_voice(name: str) -> Voice | None:
    for v in list_voice_dirs():
        if v.name == name:
            return v
    return None


def load_audio_any(raw: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return mono.astype(np.float32), target_sr


# --------------------------------------------------------------- sessions

def new_session_name() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def session_dir(name: str | None) -> Path:
    """Resolve (or create) a session folder under data/audio/sessions.
    Names are sanitized to a filesystem- and URL-safe form."""
    if name:
        safe = re.sub(r"[^A-Za-z0-9 _-]+", "_", name).strip().replace(" ", "_")[:60]
    else:
        safe = ""
    if not safe:
        safe = new_session_name()
    d = SESSIONS_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_history(limit_sessions: int = 100) -> list[dict]:
    """Sessions (newest first) with their clips (newest first)."""
    sessions = []
    for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        clips = []
        for wav in sorted(d.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = {}
            sidecar = wav.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            import soundfile as _sf

            try:
                dur = _sf.info(wav).frames / _sf.info(wav).samplerate
            except Exception:
                dur = 0.0
            clips.append({
                "file": wav.name,
                "url": f"/audio/sessions/{d.name}/{wav.name}",
                "voice": meta.get("voice", "?"),
                "language": meta.get("language", "?"),
                "chars": meta.get("chars", 0),
                "duration": round(dur, 1),
                "created": int(wav.stat().st_mtime),
            })
        sessions.append({
            "name": d.name,
            "created": int(d.stat().st_mtime),
            "clips": clips,
        })
        if len(sessions) >= limit_sessions:
            break
    return sessions


# ---------------------------------------------------------------- engines

ENGINES: dict[str, dict] = {
    "multilingual": {"state": "pending", "error": None, "model": None,
                     "t3_half": False, "t3_on_gpu": False, "gpu_enabled": True},
    "v3": {"state": "pending", "error": None, "model": None,
           "t3_half": False, "t3_on_gpu": False, "gpu_enabled": True},
}
ACTIVE_ENGINE: str | None = None
LOAD_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()

# Voice conditionals are deterministic per (engine, clip) and expensive to
# build, so cache them. Key: (engine_name, wav_path) -> (mtime, Conditionals)
CONDS_CACHE: dict[tuple[str, str], tuple[float, object]] = {}
CONDS_LOCK = threading.Lock()


def _t3_dtype_device(model):
    p = next(model.t3.parameters())
    return p.dtype, p.device


def _clone_conds(conds):
    """Clone a Conditionals object. copy.deepcopy() fails on non-leaf tensors
    (produced by s3gen.embed_ref), so clone each tensor explicitly."""
    t3_kwargs = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in conds.t3.__dict__.items()
    }
    gen_clone = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in conds.gen.items()
    }
    return conds.__class__(conds.t3.__class__(**t3_kwargs), gen_clone)


def _conds_for(engine_name: str, model, wav_path: Path):
    key = (engine_name, str(wav_path))
    mtime = wav_path.stat().st_mtime
    with CONDS_LOCK:
        hit = CONDS_CACHE.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    model.prepare_conditionals(str(wav_path), exaggeration=0.5)
    conds = _clone_conds(model.conds)
    with CONDS_LOCK:
        CONDS_CACHE[key] = (mtime, conds)
    return conds


def _assign_conds(engine_name: str, model, wav_path: Path) -> None:
    conds = _clone_conds(_conds_for(engine_name, model, wav_path))
    dtype, _dev = _t3_dtype_device(model)
    conds.t3.to(device=model.device, dtype=dtype)
    model.conds = conds


def _patch_s3gen_bridge(model, trim_tail_noise: bool) -> None:
    """Bridge device boundary (t3 may emit CUDA tokens while s3gen is on CPU)
    and optionally trim the final token's audio: it is emitted just before EOS
    with degraded attention and decodes to ~40 ms of noise (the fix applied in
    Resemble's official V3 demo code)."""
    if getattr(model, "_s3gen_patched", False):
        return
    orig_inference = model.s3gen.inference
    samples_per_token = AUDIO_SR // S3_TOKEN_RATE  # 960

    def inference_bridged(speech_tokens=None, ref_dict=None, **kw):
        if torch.is_tensor(speech_tokens):
            speech_tokens = speech_tokens.cpu()
        out_wavs, sources = orig_inference(speech_tokens=speech_tokens, ref_dict=ref_dict, **kw)
        if trim_tail_noise and speech_tokens is not None:
            n = int(speech_tokens.shape[-1])
            if n > 1:
                out_wavs = out_wavs[:, : (n - 1) * samples_per_token]
        return out_wavs, sources

    model.s3gen.inference = inference_bridged
    model._s3gen_patched = True


def _patch_fp16(model) -> None:
    """Keep conditioning dtype/device-consistent with the fp16 t3 backbone.

    1) prepare_conditionals runs wholly on CPU (it feeds the fp32 s3gen/ve
       internally); afterwards the t3 conditioning tensors are cast to t3's
       live dtype/device. conds.gen stays CPU fp32 - the fp32 s3gen consumes it.
    2) generate() pre-syncs emotion_adv to the exact t3-dtype value of the
       requested exaggeration, so upstream's 'rebuild T3Cond' branch (which
       would create a fresh fp32 tensor) never triggers.
    """
    orig_prepare = model.prepare_conditionals

    def prepare_cast(wav_fpath, exaggeration=0.5):
        real_device = model.device
        model.device = "cpu"
        try:
            orig_prepare(wav_fpath, exaggeration=exaggeration)
        finally:
            model.device = real_device
        dtype, _dev = _t3_dtype_device(model)
        model.conds.t3.to(device=model.device, dtype=dtype)

    model.prepare_conditionals = prepare_cast

    orig_generate = model.generate

    def generate_synced(text, *args, **kw):
        dtype, _dev = _t3_dtype_device(model)
        if "exaggeration" in kw:
            exag = torch.tensor(float(kw["exaggeration"]), dtype=dtype).item()
            kw["exaggeration"] = exag
        if model.conds is not None:
            model.conds.t3.emotion_adv = torch.full(
                (1, 1, 1), kw.get("exaggeration", 0.5), dtype=dtype, device=model.device
            )
        return orig_generate(text, *args, **kw)

    model.generate = generate_synced


def _try_generate(model, text: str, params: dict, language: str):
    return model.generate(
        text,
        language_id=language,
        exaggeration=params["exaggeration"],
        cfg_weight=params["cfg_weight"],
        temperature=params["temperature"],
    )


def _activate_engine(name: str) -> None:
    """Ensure this engine's t3 is the one on the GPU (2 GB card: only one
    fp16 backbone fits at a time). s3gen/ve stay on CPU regardless.
    No-op for engines banned from the GPU (CPU mode or post-OOM)."""
    global ACTIVE_ENGINE
    eng = ENGINES[name]
    if not CUDA_AVAILABLE or not eng["gpu_enabled"]:
        return
    if eng["t3_on_gpu"] and ACTIVE_ENGINE == name:
        return
    for other_name, other in ENGINES.items():
        if other_name != name and other["t3_on_gpu"] and other["model"] is not None:
            other["model"].t3.to("cpu")
            other["t3_on_gpu"] = False
    model = eng["model"]
    model.t3.to("cuda")
    eng["t3_on_gpu"] = True
    model.device = "cuda"
    ACTIVE_ENGINE = name
    torch.cuda.empty_cache()


def _engine_to_cpu(name: str) -> None:
    """Move an engine wholly to CPU fp32 (OOM fallback path)."""
    eng = ENGINES[name]
    model = eng["model"]
    model.t3.float()
    model.t3.to("cpu")
    eng["t3_on_gpu"] = False
    model.device = "cpu"


def _generate_on_engine(engine_name: str, text: str, ref_path: Path, params: dict, language: str) -> np.ndarray:
    eng = ENGINES[engine_name]
    model = eng["model"]
    _activate_engine(engine_name)
    _assign_conds(engine_name, model, ref_path)
    set_seed()
    try:
        wav = _try_generate(model, text, params, language)
        return wav.squeeze(0).float().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log(f"[{engine_name}] CUDA OOM -> retrying this chunk on CPU")
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        log(f"[{engine_name}] CUDA runtime error -> retrying this chunk on CPU")
    _engine_to_cpu(engine_name)
    eng["gpu_enabled"] = False  # stay on CPU for the rest of the session
    _assign_conds(engine_name, model, ref_path)  # re-cast conds to cpu/fp32
    set_seed()
    wav = _try_generate(model, text, params, language)
    return wav.squeeze(0).float().cpu().numpy()


def generate_chunk(text: str, ref_path: Path, params: dict, language: str, hindi: bool) -> np.ndarray:
    engine = pick_engine(hindi)
    if ENGINES[engine]["state"] != "ready":
        other = "multilingual" if engine == "v3" else "v3"
        if ENGINES[other]["state"] == "ready" and (other == "multilingual" or hindi is False):
            engine = other
        else:
            raise HTTPException(503, f"Engine '{engine}' not ready ({ENGINES[engine]['state']}). "
                                     f"{ENGINES[engine]['error'] or ''}")
    with _GEN_LOCK:
        try:
            return _generate_on_engine(engine, text, ref_path, params, language)
        except HTTPException:
            raise
        except Exception as exc:
            # non-OOM GPU failure -> full CPU retry on the same engine
            if ENGINES[engine]["t3_on_gpu"]:
                log(f"[{engine}] chunk failed on cuda ({type(exc).__name__}: {exc}); retrying on CPU")
                _engine_to_cpu(engine)
                model = ENGINES[engine]["model"]
                _assign_conds(engine, model, ref_path)
                set_seed()
                wav = _try_generate(model, text, params, language)
                return wav.squeeze(0).float().cpu().numpy()
            raise


def pick_engine(hindi: bool) -> str:
    if hindi:
        return "v3" if ENGINES["v3"]["state"] == "ready" else "multilingual"
    return "multilingual" if ENGINES["multilingual"]["state"] == "ready" else "v3"


def generate_to_wav(text: str, voice: Voice, params: dict, language: str, hindi: bool,
                    session: str | None = None) -> tuple[Path, str]:
    text = punc_norm_ml(text)
    chunks = split_text(text)
    if not chunks:
        raise HTTPException(400, "Text is empty.")
    pieces: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            pieces.append(np.zeros(int(0.12 * AUDIO_SR), dtype=np.float32))
        pieces.append(generate_chunk(c, voice.wav_path, params, language, hindi))
    audio = np.concatenate(pieces)
    sdir = session_dir(session)
    out = sdir / f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
    sf.write(out, audio, AUDIO_SR, subtype="PCM_16")
    return out, sdir.name


# ------------------------------------------------------------ engine loading

def _import_multilingual():
    from chatterbox import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS


def _import_v3():
    """Import the vendored V3 package under the 'chatterbox' name, grab the
    classes we need, then restore the normal 'chatterbox' module mapping."""
    v3_path = str(ROOT / "v3_engine")
    saved_modules = {k: v for k, v in sys.modules.items()
                     if k == "chatterbox" or k.startswith("chatterbox.")}
    for k in list(saved_modules):
        del sys.modules[k]
    sys.path.insert(0, v3_path)
    try:
        from chatterbox.tts import ChatterboxTTS as V3TTS, Conditionals as V3Conditionals
        from chatterbox.models.t3 import T3 as V3T3
        from chatterbox.models.t3.modules.t3_config import T3ConfigMultilingual
        from chatterbox.models.s3gen import S3Gen as V3S3Gen
        from chatterbox.models.voice_encoder import VoiceEncoder as V3VoiceEncoder
        from chatterbox.models.tokenizers import MTLTokenizer as V3MTLTokenizer

        return {
            "TTS": V3TTS, "Conditionals": V3Conditionals, "T3": V3T3,
            "T3ConfigMultilingual": T3ConfigMultilingual, "S3Gen": V3S3Gen,
            "VoiceEncoder": V3VoiceEncoder, "MTLTokenizer": V3MTLTokenizer,
        }
    finally:
        sys.path.remove(v3_path)
        for k in [k for k in sys.modules if k == "chatterbox" or k.startswith("chatterbox.")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)


def _build_v3_model():
    """Assemble the Hindi V3 engine exactly like the official demo Space:
    shared assets (ve.pt, conds.pt) from the base repo cache, finetune assets
    (t3_hi.safetensors, s3gen_v3.pt, tokenizer json) from the local clone."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors

    C = _import_v3()

    base_dir = Path(hf_hub_download(repo_id="ResembleAI/chatterbox", filename="ve.pt")).parent
    ft_dir = ROOT / "Chatterbox-Multilingual-hi"
    t3_file = ft_dir / "t3_hi.safetensors"
    s3gen_file = ft_dir / "s3gen_v3.pt"
    if not t3_file.exists() or t3_file.stat().st_size < 1e9:
        raise FileNotFoundError(f"{t3_file} missing or still an LFS stub ({t3_file.stat().st_size if t3_file.exists() else 0} bytes)")
    if not s3gen_file.exists() or s3gen_file.stat().st_size < 1e9:
        raise FileNotFoundError(f"{s3gen_file} missing or still an LFS stub")

    ve = C["VoiceEncoder"]()
    ve.load_state_dict(torch.load(base_dir / "ve.pt", weights_only=True, map_location="cpu"))
    ve.eval()

    t3_cfg = C["T3ConfigMultilingual"]()
    t3_cfg.text_tokens_dict_size = 2454
    t3 = C["T3"](t3_cfg)
    t3_state = load_safetensors(t3_file)
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.eval()

    s3gen = C["S3Gen"]()
    s3gen.load_state_dict(torch.load(s3gen_file, weights_only=True, map_location="cpu"), strict=False)
    s3gen.eval()

    tokenizer = C["MTLTokenizer"](str(ft_dir / "grapheme_mtl_merged_expanded_v1.json"))

    conds = None
    if (builtin_voice := base_dir / "conds.pt").exists():
        conds = C["Conditionals"].load(builtin_voice).to("cpu")

    model = C["TTS"](t3, s3gen, ve, tokenizer, "cpu", conds=conds)
    model._t3_half = False
    return model


def _probe(model, engine_name: str, on_gpu: bool) -> None:
    probe_ref = PRESETS_DIR / "atlas.wav"
    if not probe_ref.exists():
        refs = sorted(PRESETS_DIR.glob("*.wav"))
        if not refs:
            raise RuntimeError("no reference clip available for probe")
        probe_ref = refs[0]
    params = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}
    if on_gpu:
        _activate_engine(engine_name)
    else:
        model.device = "cpu"
    _assign_conds(engine_name, model, probe_ref)
    set_seed()
    wav = _try_generate(model, "Hello.", params, "en")
    assert float(wav.abs().max()) > 0


def _load_multilingual_engine() -> None:
    eng = ENGINES["multilingual"]
    eng["state"] = "loading"
    t0 = time.time()
    try:
        cls = _import_multilingual()
        log("[ml] loading multilingual weights (CPU)...")
        model = cls.from_pretrained("cpu")
        log(f"[ml] weights ready in {time.time() - t0:.1f}s, placing...")
        eng["model"] = model

        if CUDA_AVAILABLE and GPU_OK:
            model._t3_half = True
            model.t3.half()
            _patch_fp16(model)
            _patch_s3gen_bridge(model, trim_tail_noise=True)
            try:
                _probe(model, "multilingual", on_gpu=True)
                eng.update(state="ready", t3_half=True, error=None)
                log(f"[ml] READY on cuda (t3-fp16, tail-noise-trim) in {time.time() - t0:.1f}s")
                return
            except Exception as exc:
                log(f"[ml] cuda probe failed ({type(exc).__name__}: {exc}); falling back to CPU")
                torch.cuda.empty_cache()
                _engine_to_cpu("multilingual")
                eng["gpu_enabled"] = False
        model.t3.float()
        _patch_s3gen_bridge(model, trim_tail_noise=True)
        _probe(model, "multilingual", on_gpu=False)
        eng.update(state="ready", t3_half=False, error=None)
        log(f"[ml] READY on cpu in {time.time() - t0:.1f}s")
    except Exception as exc:
        eng.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        log(f"[ml] FAILED: {eng['error']}")


def _load_v3_engine() -> None:
    eng = ENGINES["v3"]
    eng["state"] = "loading"
    t0 = time.time()
    try:
        log("[v3] loading Hindi finetune weights (CPU)...")
        model = _build_v3_model()
        log(f"[v3] weights ready in {time.time() - t0:.1f}s, placing...")
        eng["model"] = model

        if CUDA_AVAILABLE and GPU_OK:
            model._t3_half = True
            model.t3.half()
            _patch_fp16(model)
            _patch_s3gen_bridge(model, trim_tail_noise=False)  # v3 generate() trims internally
            try:
                _probe(model, "v3", on_gpu=True)
                eng.update(state="ready", t3_half=True, error=None)
                log(f"[v3] READY on cuda (t3-fp16) in {time.time() - t0:.1f}s")
                return
            except Exception as exc:
                log(f"[v3] cuda probe failed ({type(exc).__name__}: {exc}); falling back to CPU")
                torch.cuda.empty_cache()
                _engine_to_cpu("v3")
                eng["gpu_enabled"] = False
        model.t3.float()
        _patch_s3gen_bridge(model, trim_tail_noise=False)
        _probe(model, "v3", on_gpu=False)
        eng.update(state="ready", t3_half=False, error=None)
        log(f"[v3] READY on cpu in {time.time() - t0:.1f}s")
    except Exception as exc:
        eng.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        log(f"[v3] FAILED: {eng['error']}")


def _loader_runner() -> None:
    with LOAD_LOCK:
        if ENGINES["multilingual"]["state"] == "pending":
            _load_multilingual_engine()
        if ENGINES["v3"]["state"] == "pending":
            _load_v3_engine()
        # park the GPU on the primary engine when both are up
        if ENGINES["multilingual"]["state"] == "ready":
            try:
                _activate_engine("multilingual")
            except Exception as exc:
                log(f"gpu re-activation failed: {exc}")


def load_engines_async() -> None:
    threading.Thread(target=_loader_runner, daemon=True).start()


# ------------------------------------------------------------------ app

app = FastAPI(title="Chatterbox TTS Studio")


class GenerateRequest(BaseModel):
    text: str
    voice: str
    language: str = "auto"  # "en" | "hi" | "auto"
    session: str = ""       # optional session folder name; empty = new timestamped one
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8


@app.on_event("startup")
def _startup() -> None:
    if not list_voice_dirs():
        log("no voice presets found - run scripts/make_presets.py to bootstrap them")
    load_engines_async()


@app.get("/api/status")
def status():
    engines = {
        name: {"state": eng["state"], "error": eng["error"],
               "t3_on_gpu": eng["t3_on_gpu"], "t3_half": eng["t3_half"]}
        for name, eng in ENGINES.items()
    }
    ready_engines = [n for n, e in ENGINES.items() if e["state"] == "ready"]
    return {
        "cuda": CUDA_AVAILABLE,
        "gpu_free_gb": round(GPU_FREE_GB, 2),
        "ready": bool(ready_engines),
        "device": "cuda" if any(ENGINES[n]["t3_on_gpu"] for n in ready_engines) else
                  ("cpu" if ready_engines else None),
        "half": any(ENGINES[n]["t3_half"] for n in ready_engines),
        "state": "ready" if ready_engines else
                 ("failed" if all(e["state"] == "failed" for e in ENGINES.values()) else "loading"),
        "error": next((e["error"] for e in ENGINES.values() if e["error"]), None),
        "voice_count": len(list_voice_dirs()),
        "engines": engines,
    }


@app.get("/api/voices")
def voices():
    return [v.to_dict() for v in list_voice_dirs()]


@app.post("/api/voices")
async def create_voice(name: str, file: UploadFile = File(...)):
    if not any(e["state"] == "ready" for e in ENGINES.values()):
        raise HTTPException(503, "Model is still loading, try again shortly.")
    voice_name = _safe_name(name)
    if get_voice(voice_name):
        raise HTTPException(409, f"Voice '{voice_name}' already exists.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload.")
    try:
        audio, sr = load_audio_any(raw, AUDIO_SR)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    if len(audio) < AUDIO_SR:
        raise HTTPException(400, "Clip too short - use at least 2-3 seconds of clean speech.")
    audio = audio[: MAX_UPLOAD_SECONDS * AUDIO_SR]
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak * 0.95).astype(np.float32)
    dest = USER_VOICES_DIR / f"{voice_name}.wav"
    sf.write(dest, audio, sr, subtype="PCM_16")
    return {"ok": True, "voice": Voice(voice_name, dest, "custom").to_dict()}


@app.delete("/api/voices/{name}")
def delete_voice(name: str):
    v = get_voice(name)
    if v is None:
        raise HTTPException(404, "Voice not found.")
    if v.kind == "preset":
        raise HTTPException(400, "Built-in presets cannot be deleted.")
    v.wav_path.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not any(e["state"] == "ready" for e in ENGINES.values()):
        raise HTTPException(503, "Model not ready yet - still loading.")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Enter some text first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text too long (max {MAX_TEXT_CHARS} chars).")
    voice = get_voice(req.voice)
    if voice is None:
        raise HTTPException(404, f"Unknown voice '{req.voice}'.")

    hindi = has_devanagari(text) if req.language == "auto" else req.language == "hi"
    language = "hi" if hindi else "en"

    params = {
        "exaggeration": float(np.clip(req.exaggeration, 0.25, 2.0)),
        "cfg_weight": float(np.clip(req.cfg_weight, 0.0, 1.0)),
        "temperature": float(np.clip(req.temperature, 0.05, 2.0)),
    }
    t0 = time.time()
    try:
        out, session_name = generate_to_wav(text, voice, params, language, hindi, req.session)
    except HTTPException:
        raise
    except Exception as exc:
        log(f"generation failed: {exc}")
        raise HTTPException(500, f"Generation failed: {exc}") from exc
    # sidecar metadata for the history view
    try:
        Path(str(out).replace(".wav", ".json")).write_text(
            json.dumps({
                "voice": voice.to_dict()["name"],
                "language": language,
                "chars": len(text),
                "text": text[:200],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    engine_used = "v3" if (hindi and ENGINES["v3"]["state"] == "ready") else "multilingual"
    log(f"generated {len(text)} chars [{language}] with '{voice.name}' via {engine_used} "
        f"in {time.time() - t0:.1f}s -> session {session_name}")
    return FileResponse(
        out,
        media_type="audio/wav",
        filename=out.name,
        headers={"X-Session": session_name},
    )


@app.get("/api/history")
def history():
    return list_history()


@app.get("/api/gpu")
def gpu_usage():
    """Live GPU utilization + VRAM (pynvml; cheap enough to poll every ~2 s)."""
    if not CUDA_AVAILABLE:
        return {"available": False}
    info: dict = {"available": True}
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        info["vram_used_gb"] = round((total_b - free_b) / 1024**3, 2)
        info["vram_total_gb"] = round(total_b / 1024**3, 2)
    except Exception:
        pass
    if _NVML_HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            info["util_percent"] = int(util.gpu)
            info["mem_percent"] = int(util.memory)
            temp = pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            info["temp_c"] = int(temp)
        except Exception:
            pass
    return info


# Mount order matters: most-specific first, catch-all UI last.
# Voice previews live in data/voices, generated clips in data/audio/sessions.
app.mount("/audio/voices", StaticFiles(directory=VOICES_DIR), name="voices")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8488, log_level="info")
