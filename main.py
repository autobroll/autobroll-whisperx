# main.py
import os
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple
import multiprocessing as mp
import threading

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import ffmpeg
import aiohttp
import traceback

import whisperx  # align + diarization
try:
    from faster_whisper import WhisperModel  # ASR direct, sans VAD
    _FW_OK = True
except Exception:
    _FW_OK = False

# -------------------- Config --------------------
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v2")

DEVICE_ENV = os.getenv("WHISPERX_DEVICE", "").strip().lower()
if DEVICE_ENV == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
elif DEVICE_ENV in ("cuda", "cpu"):
    DEVICE = DEVICE_ENV
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

ALIGN_DEVICE_ENV = os.getenv("WHISPERX_ALIGN_DEVICE", "").strip().lower()
ALIGN_DEVICE = ALIGN_DEVICE_ENV if ALIGN_DEVICE_ENV in ("cuda", "cpu") else "cpu"

BEAM_SIZE = int(os.getenv("WHISPERX_BEAM_SIZE", "5"))
VAD_FILTER = os.getenv("WHISPERX_VAD_FILTER", "false").lower() == "true"

DIARIZATION = os.getenv("WHISPERX_DIARIZATION", "false").lower() == "true"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN", ""))

API_KEY = os.getenv("API_KEY", "")

DEVICE_INDEX = int(os.getenv("WHISPERX_DEVICE_INDEX", "0"))  # utile sur certaines stacks

# Timeout (sec) pour tuer proprement le sous-processus GPU si blocage
ASR_GPU_TIMEOUT = int(os.getenv("WHISPERX_ASR_GPU_TIMEOUT", "180"))

app = FastAPI(title="whisperx-api", version="1.2.0")

_models_cache: Dict[str, Any] = {
    "asr_fw_cpu": None,         # WhisperModel CPU
    "align": {},                # language -> (align_model, metadata)
    "diar": None
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
                raise HTTPException(status_code=400, detail=f"Failed to fetch media: HTTP {resp.status}")
            with open(tmp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    f.write(chunk)
    return tmp_path

def _extract_audio_16k_mono(in_path: str) -> str:
    out_fd, out_path = tempfile.mkstemp(suffix=".wav"); os.close(out_fd)
    try:
        (ffmpeg.input(in_path)
               .output(out_path, ac=1, ar="16000", f="wav", vn=None, loglevel="error")
               .overwrite_output()
               .run())
    except ffmpeg.Error as e:
        try: os.unlink(out_path)
        except: pass
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e}")
    return out_path

def _to_srt(segments: List[Dict[str, Any]]) -> str:
    def ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    out = []
    for i, seg in enumerate(segments, 1):
        start = float(seg["start"]); end = float(seg["end"]); text = seg.get("text", "").strip()
        out.append(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n")
    return "\n".join(out).strip() + "\n"

def _to_vtt(segments: List[Dict[str, Any]]) -> str:
    def ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    out = ["WEBVTT\n"]
    for i, seg in enumerate(segments, 1):
        start = float(seg["start"]); end = float(seg["end"]); text = seg.get("text", "").strip()
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
            if w.get("word") is None: continue
            s["words"].append({
                "text": w.get("word"),
                "start": float(w.get("start", s["start"])) if w.get("start") is not None else None,
                "end": float(w.get("end", s["end"])) if w.get("end") is not None else None,
                "prob": float(w.get("probability", 0)) if w.get("probability") is not None else None
            })
        segs.append(s)
    return segs

# -------------------- Model helpers --------------------
def _ensure_cpu_model():
    if not _FW_OK:
        raise HTTPException(status_code=500, detail="faster-whisper indisponible")
    if _models_cache["asr_fw_cpu"] is None:
        _models_cache["asr_fw_cpu"] = WhisperModel(
            WHISPERX_MODEL, device="cpu", compute_type="int8"
        )
    return _models_cache["asr_fw_cpu"]

def _get_align_model(language: str):
    if language in _models_cache["align"]:
        return _models_cache["align"][language]
    align_model, metadata = whisperx.load_align_model(language_code=language, device=ALIGN_DEVICE)
    _models_cache["align"][language] = (align_model, metadata)
    return align_model, metadata

def _ensure_diarization():
    if not DIARIZATION:
        return None
    if _models_cache["diar"] is None:
        _models_cache["diar"] = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=ALIGN_DEVICE)
    return _models_cache["diar"]

# -------------------- GPU subprocess runner --------------------
def _gpu_asr_worker(args: Tuple[str, Optional[str], str, int, bool, int, str], pipe):
    """
    Lance l'ASR dans un sous-processus pour éviter de tuer le serveur si CUDA segfault/OOM.
    Renvoie via Pipe: {"ok": True, "segments": [...], "lang": "..."} ou {"ok": False, "error": "..."}.
    """
    wav_path, language, model_name, beam_size, vad_filter, device_index, compute_type = args
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel(
            model_name,
            device="cuda",
            compute_type=compute_type,
            device_index=device_index,
        )
        segments_iter, info = asr.transcribe(
            wav_path, language=language, vad_filter=vad_filter, beam_size=beam_size
        )
        segs = [{"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
                for s in segments_iter]
        lang = language or getattr(info, "language", "en")
        pipe.send({"ok": True, "segments": segs, "lang": lang})
    except Exception as e:
        pipe.send({"ok": False, "error": f"{type(e).__name__}: {str(e)}"})
    finally:
        try: pipe.close()
        except: pass

def _asr_transcribe_safe_gpu_then_cpu(wav_path: str, language: Optional[str]) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Tente l'ASR sur GPU dans un sous-processus. Si le sous-processus meurt/échoue/timeout,
    retombe en CPU dans le process principal. Retourne (segments, lang, debug_info).
    """
    if DEVICE == "cuda":
        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(
            target=_gpu_asr_worker,
            args=((
                wav_path, language, WHISPERX_MODEL, BEAM_SIZE, VAD_FILTER,
                DEVICE_INDEX, os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16")
            ), child_conn),
            daemon=True,
        )
        p.start()
        try:
            if parent_conn.poll(ASR_GPU_TIMEOUT):
                res = parent_conn.recv()
            else:
                res = {"ok": False, "error": f"GPU timeout after {ASR_GPU_TIMEOUT}s"}
        finally:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        if res.get("ok") is True:
            return res["segments"], res["lang"], {"engine": "gpu"}
        # GPU a échoué → on tombera en CPU
        cpu_reason = res.get("error", "gpu process failed/terminated")
    else:
        cpu_reason = "device not cuda"

    # CPU fallback (process principal)
    asr_cpu = _ensure_cpu_model()
    segments_iter, info = asr_cpu.transcribe(
        wav_path, language=language, vad_filter=VAD_FILTER, beam_size=BEAM_SIZE
    )
    segs = [{"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
            for s in segments_iter]
    lang = language or getattr(info, "language", "en")
    return segs, lang, {"engine": "cpu", "gpu_fail": cpu_reason}

# -------------------- Schemas --------------------
class TranscribeUrlIn(BaseModel):
    url: str
    language: Optional[str] = None  # e.g., "fr", "en"

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "align_device": ALIGN_DEVICE,
        "model": WHISPERX_MODEL,
        "compute_type": COMPUTE_TYPE,
        "diarization": DIARIZATION,
        "auth_required": bool(API_KEY),
        "engine": "faster-whisper + whisperx-align",
        "device_index": DEVICE_INDEX,
    }

@app.post("/warmup", dependencies=[Depends(require_api_key)])
def warmup(language: Optional[str] = "fr"):
    # Pré-charge align (CPU) et init CPU ASR (léger). GPU sera chargé au 1er call via sous-processus.
    _ensure_cpu_model()
    _get_align_model(language or "fr")
    if DIARIZATION:
        _ensure_diarization()
    return {
        "ok": True, "warmed": True, "device": DEVICE, "align_device": ALIGN_DEVICE,
        "model": WHISPERX_MODEL, "language": language or "fr",
        "engine": "faster-whisper + whisperx-align"
    }

# ---- Warmup GPU non bloquant : précharge le modèle CTranslate2 en arrière-plan
def _prefetch_gpu_model():
    try:
        from faster_whisper import WhisperModel
        WhisperModel(
            WHISPERX_MODEL,
            device="cuda",
            compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16"),
            device_index=DEVICE_INDEX,
        )
    except Exception:
        # On ignore ici pour ne pas bloquer le démarrage ; l'inférence fera le fallback si besoin
        pass

@app.post("/warmup/gpu", dependencies=[Depends(require_api_key)])
def warmup_gpu():
    if DEVICE != "cuda":
        return {"ok": False, "message": "CUDA indisponible sur ce pod"}
    threading.Thread(target=_prefetch_gpu_model, daemon=True).start()
    return {"ok": True, "started": True, "device": DEVICE, "compute_type": COMPUTE_TYPE}

# --- Transcribe endpoints (avec paramètre engine)
@app.post("/transcribe/url", dependencies=[Depends(require_api_key)])
async def transcribe_url(
    body: TranscribeUrlIn,
    engine: Optional[str] = Query("auto", description="auto|cpu|gpu")
):
    media_path = await _download_url_to_file(body.url, suffix=".bin")
    try:
        return await _process_any(media_path, language=body.language, engine=engine)
    finally:
        try: os.unlink(media_path)
        except: pass

@app.post("/transcribe/file", dependencies=[Depends(require_api_key)])
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form("auto")
):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix); os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        return await _process_any(tmp_path, language=language, engine=engine)
    finally:
        try: os.unlink(tmp_path)
        except: pass

# -------------------- Core processing --------------------
async def _process_any(in_path: str, language: Optional[str] = None, engine: str = "auto"):
    wav_path = _extract_audio_16k_mono(in_path)
    try:
        # 1) ASR
        try:
            if engine.lower() == "cpu" or DEVICE != "cuda":
                # CPU direct
                asr_cpu = _ensure_cpu_model()
                segments_iter, info = asr_cpu.transcribe(
                    wav_path, language=language, vad_filter=VAD_FILTER, beam_size=BEAM_SIZE
                )
                segments_list = [{"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
                                 for s in segments_iter]
                lang = language or getattr(info, "language", "en")
                dbg = {"engine": "cpu", "forced": engine.lower() == "cpu"}
            elif engine.lower() in ("auto", "gpu"):
                # GPU sous-processus + fallback CPU
                segments_list, lang, dbg = _asr_transcribe_safe_gpu_then_cpu(wav_path, language)
            else:
                return JSONResponse(status_code=400, content={"ok": False, "error": "engine must be auto|cpu|gpu"})
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "ok": False, "stage": "asr", "error": str(e), "trace": traceback.format_exc()
            })

        # 2) Alignement (souvent CPU)
        try:
            align_model, metadata = _get_align_model(lang)
            aligned = whisperx.align(
                segments_list, align_model, metadata, wav_path, ALIGN_DEVICE, return_char_alignments=False
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "ok": False, "stage": "whisperx.align", "error": str(e), "trace": traceback.format_exc(), "dbg": dbg
            })

        # 3) Diarisation (optionnelle)
        if DIARIZATION:
            try:
                diar = _ensure_diarization()
                if diar is not None:
                    diar_segments = diar(wav_path)
                    aligned = whisperx.assign_word_speakers(diar_segments, aligned)
            except Exception as e:
                return JSONResponse(status_code=500, content={
                    "ok": False, "stage": "diarization", "error": str(e), "trace": traceback.format_exc(), "dbg": dbg
                })

        # 4) Sorties
        merged = _merge_words_into_segments(aligned)
        vtt = _to_vtt(merged)
        srt = _to_srt(merged)

        payload = {"ok": True, "language": lang, "segments": merged, "vtt": vtt, "srt": srt}
        if dbg: payload["dbg"] = dbg
        return JSONResponse(payload)
    finally:
        try: os.unlink(wav_path)
        except: pass

if __name__ == "__main__":
    # important pour le multiprocessing sous certains runtimes
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
