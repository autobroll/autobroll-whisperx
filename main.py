# main.py
import os
import io
import json
import tempfile
import asyncio
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import numpy as np
import ffmpeg
import aiohttp

import whisperx  # pip install whisperx

# -------------------- Config --------------------
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v2")
DEVICE_ENV = os.getenv("WHISPERX_DEVICE", "").strip().lower()
DEVICE = DEVICE_ENV if DEVICE_ENV in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
DIARIZATION = os.getenv("WHISPERX_DIARIZATION", "false").lower() == "true"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN", ""))

# Désactivé par défaut pour éviter le download VAD qui plantait (HTTP 301)
VAD_ENABLED = os.getenv("WHISPERX_VAD", "false").lower() == "true"

API_KEY = os.getenv("API_KEY", "")  # ex: autobroll_secret_1

app = FastAPI(title="whisperx-api", version="1.0.0")

_models_cache = {
    "asr": None,                 # whisperx ASR model
    "asr_lang": None,           # language code detected/forced
    "align": {},                # language -> (align_model, metadata)
    "diar": None                # diarization pipeline
}

# -------------------- Auth helper --------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------- Utils --------------------
async def _download_url_to_file(url: str, suffix: str = "") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to fetch media: {resp.status}")
            with open(tmp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    f.write(chunk)
    return tmp_path

def _extract_audio_16k_mono(in_path: str) -> str:
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        (
            ffmpeg
            .input(in_path)
            .output(out_path, ac=1, ar="16000", f="wav", vn=None, loglevel="error")
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        try:
            os.unlink(out_path)
        except:
            pass
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e}")
    return out_path

def _ensure_asr_model(language: Optional[str] = None):
    # Load ASR once
    if _models_cache["asr"] is None:
        try:
            # Désactive/active le VAD via vad_options (compatible avec ta version)
            _models_cache["asr"] = whisperx.load_model(
                WHISPERX_MODEL,
                DEVICE,
                compute_type=COMPUTE_TYPE,
                vad_options={"use_vad": VAD_ENABLED}
            )
        except TypeError:
            # Fallback si la signature change : sans VAD options
            _models_cache["asr"] = whisperx.load_model(
                WHISPERX_MODEL,
                DEVICE,
                compute_type=COMPUTE_TYPE
            )
    if language:
        _models_cache["asr_lang"] = language

def _get_align_model(language: str):
    if language in _models_cache["align"]:
        return _models_cache["align"][language]
    align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
    _models_cache["align"][language] = (align_model, metadata)
    return align_model, metadata

def _ensure_diarization():
    if not DIARIZATION:
        return None
    if _models_cache["diar"] is None:
        _models_cache["diar"] = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    return _models_cache["diar"]

def _to_srt(segments: List[Dict[str, Any]]) -> str:
    def ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        start = float(seg["start"]); end = float(seg["end"])
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n")
    return "\n".join(lines).strip() + "\n"

def _to_vtt(segments: List[Dict[str, Any]]) -> str:
    def ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    out = ["WEBVTT\n"]
    for i, seg in enumerate(segments, 1):
        start = float(seg["start"]); end = float(seg["end"])
        text = seg.get("text", "").strip()
        out.append(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n")
    return "\n".join(out).strip() + "\n"

def _merge_words_into_segments(aligned_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = []
    for seg in aligned_result.get("segments", []):
        s = {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip(),
            "words": []
        }
        for w in seg.get("words", []) or []:
            if w.get("word") is None:
                continue
            s["words"].append({
                "text": w.get("word"),
                "start": float(w.get("start", s["start"])) if w.get("start") is not None else None,
                "end": float(w.get("end", s["end"])) if w.get("end") is not None else None,
                "prob": float(w.get("probability", 0)) if w.get("probability") is not None else None
            })
        segs.append(s)
    return segs

# -------------------- Schemas --------------------
class TranscribeUrlIn(BaseModel):
    url: str
    language: Optional[str] = None  # e.g. "fr", "en", etc.

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "model": WHISPERX_MODEL,
        "compute_type": COMPUTE_TYPE,
        "diarization": DIARIZATION,
        "vad_enabled": VAD_ENABLED,
        "auth_required": bool(API_KEY),
    }

@app.post("/transcribe/url", dependencies=[Depends(require_api_key)])
async def transcribe_url(body: TranscribeUrlIn):
    media_path = await _download_url_to_file(body.url, suffix=".bin")
    try:
        return await _process_any(media_path, language=body.language)
    finally:
        try:
            os.unlink(media_path)
        except:
            pass

@app.post("/transcribe/file", dependencies=[Depends(require_api_key)])
async def transcribe_file(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        return await _process_any(tmp_path, language=language)
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# -------------------- Core processing --------------------
async def _process_any(in_path: str, language: Optional[str] = None):
    wav_path = _extract_audio_16k_mono(in_path)

    try:
        _ensure_asr_model(language=language)

        # 1) ASR
        asr_model = _models_cache["asr"]
        result = asr_model.transcribe(wav_path, language=language)

        # Detect language if not provided
        lang = language or result.get("language", "en")

        # 2) Alignment
        align_model, metadata = _get_align_model(lang)
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            wav_path,
            DEVICE,
            return_char_alignments=False
        )

        # 3) Diarization (optional)
        if DIARIZATION:
            diar = _ensure_diarization()
            if diar is not None:
                diar_segments = diar(wav_path)
                aligned = whisperx.assign_word_speakers(diar_segments, aligned)

        # 4) Build outputs
        segments = _merge_words_into_segments(aligned)
        vtt = _to_vtt(segments)
        srt = _to_srt(segments)

        return JSONResponse({
            "ok": True,
            "language": lang,
            "segments": segments,
            "vtt": vtt,
            "srt": srt
        })
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
