from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download, hf_hub_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3ConfigMultilingual
from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/Chatterbox-Multilingual-hi"
BASE_REPO_ID = "ResembleAI/chatterbox"
S3GEN_FILENAME = "s3gen_v3.pt"
T3_FILENAME = "t3_hi.safetensors"
TOKENIZER_FILENAME = "grapheme_mtl_merged_expanded_v1.json"
T3_TEXT_VOCAB_SIZE = 2454


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    """
    Language-agnostic text-to-speech model that generates high-quality speech from text in any language.
    This model uses English tokenization internally and accepts text in any language without requiring language specification.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    def to(self, device):
        self.t3.to(device).eval()
        self.s3gen.to(device).eval()
        self.ve.to(device).eval()
        if self.conds is not None:
            self.conds = self.conds.to(device)
        self.device = str(device)
        return self



    @classmethod
    def from_local(cls, ckpt_dir, device, t3_filename: str = None, s3gen_filename: str = None) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)
        t3_filename = t3_filename or T3_FILENAME
        s3gen_filename = s3gen_filename or S3GEN_FILENAME

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location="cpu")
        )
        ve.to(device).eval()

        t3_cfg = T3ConfigMultilingual()
        t3_cfg.text_tokens_dict_size = T3_TEXT_VOCAB_SIZE
        t3 = T3(t3_cfg)
        t3_state = load_safetensors(ckpt_dir / t3_filename)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / s3gen_filename, weights_only=True, map_location="cpu"),
            strict=False,
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / TOKENIZER_FILENAME)
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxTTS':
        token = os.getenv("HF_TOKEN")
        base_files = ["ve.pt", TOKENIZER_FILENAME, "conds.pt"]
        base_dir = Path(
            snapshot_download(
                repo_id=BASE_REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=base_files + [T3_FILENAME, S3GEN_FILENAME] if REPO_ID == BASE_REPO_ID else base_files,
                token=token,
            )
        )

        if REPO_ID == BASE_REPO_ID:
            ckpt_dir = base_dir
        else:
            for filename in (T3_FILENAME, S3GEN_FILENAME):
                src_path = Path(hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type="model",
                    token=token,
                ))
                link = base_dir / filename
                if not link.exists():
                    try:
                        os.symlink(src_path, link)
                    except OSError:
                        import shutil as _sh
                        _sh.copy(src_path, link)
            ckpt_dir = base_dir
        return cls.from_local(ckpt_dir, device, t3_filename=T3_FILENAME, s3gen_filename=S3GEN_FILENAME)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        language_id="en",
    ):
        """
        Generate speech from text using the language-agnostic model.
        
        Args:
            text (str): Text to synthesize into speech (supports any language)
            audio_prompt_path (str, optional): Path to reference audio for voice cloning
            exaggeration (float): Controls speech expressiveness (0.25-2.0)
            cfg_weight (float): CFG weight for generation guidance
            temperature (float): Controls randomness in generation
            
        Returns:
            torch.Tensor: Generated audio waveform
        """
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        lang = (language_id or "en").lower() if language_id else None
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            # Drop the final speech token's audio: it is emitted just before
            # EOS with degraded attention and decodes to ~40 ms of noise.
            n_tokens = int(speech_tokens.shape[-1])
            st_len = max(1, n_tokens - 1)
            wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]

            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
