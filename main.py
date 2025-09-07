# main.py
import os
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple
import multiprocessing as mp
import threading

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# ⚠️ Import lourds (torch / whisperx / faster_whisper) retirés du top-level
# pour éviter tout crash avant que /ping réponde.
import ffmpeg
import aiohttp
import traceback
import importlib

# -------------------- Config --------------------
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v2")

# Ne pas importer torch ici. On respecte simplement la config.
DEVICE_ENV = os.getenv("WHISPERX_DEVICE", "").strip().lower()
DEVICE = DEVICE_ENV if DEVICE_ENV in ("cuda", "cpu") else "cuda"

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

app = FastAPI(title="whisperx-api", version="1.2.2")

_models_cache: Dict[str, Any] = {
    "asr_fw_cpu": None,         # WhisperModel CPU
    "align": {},                # language -> (align_model, metadata)
    "diar": None
}

# Lazy modules (chargés à la demande)
_lazy_modules: Dict[str, Any] = {
    "whisperx": None,
    "faster_whisper": None,
}

def _wx():
    """Lazy import de whisperx."""
    if _lazy_modules["whisperx"] is None:
        _lazy_modules["whisperx"] = importlib.import_module("whisperx")
    return _lazy_modules["whisperx"]

def _fw():
    """Lazy import de faster_whisper."""
    if _lazy_modules["faster_whisper"] is None:
        _lazy_modules["faster_whisper"] = importlib.import_module("faster_whisper")
    return _lazy_modules["faster_whisper"]

# -------------------- Warmup flags --------------------
WARMED_CPU = False
WARMED_GPU = False
WARMUP_LOCK = threading.Lock()

# -------------------- Auth helper --------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------- Liveness / Health for Runpod --------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    # Petit endpoint pour tester depuis le navigateur
    return "whisperx-api ok"

@app.get("/ping", response_class=PlainTextResponse)
def ping():
    # Ultra rapide: utilisé par le Load Balancer (PORT_HEALTH)
    return "pong"

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
        "warmed_cpu": WARMED_CPU,
        "warmed_gpu": WARMED_GPU,
    }

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
    try:
        WhisperModel = _fw().WhisperModel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"faster-whisper indisponible: {e}")
    if _models_cache["asr_fw_cpu"] is None:
        _models_cache["asr_fw_cpu"] = WhisperModel(
            WHISPERX_MODEL, device="cpu", compute_type="int8"
        )
    return _models_cache["asr_fw_cpu"]

def _get_align_model(language: str):
    if language in _models_cache["align"]:
        return _models_cache["align"][language]
    wx = _wx()
    align_model, metadata = wx.load_align_model(language_code=language, device=ALIGN_DEVICE)
    _models_cache["align"][language] = (align_model, metadata)
    return align_model, metadata

def _ensure_diarization():
    if not DIARIZATION:
        return None
    if _models_cache["diar"] is None:
        wx = _wx()
        _models_cache["diar"] = wx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=ALIGN_DEVICE)
    return _models_cache["diar"]

# -------------------- Warmup helpers --------------------
def _warmup_cpu():
    """Précharge aligner (en, fr) + diar (si activée) en non-bloquant."""
    global WARMED_CPU
    with WARMUP_LOCK:
        if WARMED_CPU:
            return
        # asr cpu (utile si fallback)
        _ensure_cpu_model()
        # aligners
        for _lang in ("fr", "en"):
            try:
                _get_align_model(_lang)
            except Exception as e:
                print(f"[warmup-cpu] align {_lang} -> {e}")
        # diarisation (optionnelle)
        if DIARIZATION:
            try:
                _ensure_diarization()
            except Exception as e:
                print(f"[warmup-cpu] diar -> {e}")
        WARMED_CPU = True
        print("[warmup-cpu] done")

# -------------------- GPU subprocess runner --------------------
def _gpu_asr_worker(args: Tuple[str, Optional[str], str, int, bool, int, str], pipe):
    wav_path, language, model_name, beam_size, vad_filter, device_index, compute_type = args
    try:
        from faster_whisper import WhisperModel  # import local au worker
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
    if DEVICE == "cuda":
        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(
            target=_gpu_asr_worker,
            args=((wav_path, language, WHISPERX_MODEL, BEAM_SIZE, VAD_FILTER,
                   DEVICE_INDEX, os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16")), child_conn),
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
        cpu_reason = res.get("error", "gpu process failed/terminated")
    else:
        cpu_reason = "device not cuda"

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
@app.post("/warmup", dependencies=[Depends(require_api_key)])
def warmup(language: Optional[str] = "fr"):
    # lance le warmup CPU en thread (non bloquant)
    threading.Thread(target=_warmup_cpu, daemon=True).start()
    return {
        "ok": True, "started": True, "device": DEVICE, "align_device": ALIGN_DEVICE,
        "model": WHISPERX_MODEL, "language": language or "fr",
        "engine": "faster-whisper + whisperx-align",
        "warmed_cpu": WARMED_CPU, "warmed_gpu": WARMED_GPU
    }

def _prefetch_gpu_model():
    """Charge le modèle ASR GPU pour éliminer la latence du 1er appel."""
    global WARMED_GPU
    try:
        WhisperModel = _fw().WhisperModel
        WhisperModel(
            WHISPERX_MODEL,
            device="cuda",
            compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8_float16"),
            device_index=DEVICE_INDEX,
        )
        WARMED_GPU = True
        print("[warmup-gpu] done")
    except Exception as e:
        print("[warmup-gpu] failed:", e)

@app.post("/warmup/gpu", dependencies=[Depends(require_api_key)])
def warmup_gpu():
    if DEVICE != "cuda":
        return {"ok": False, "message": "CUDA indisponible sur ce pod"}
    # déclenche CPU + GPU warmups en tâche de fond
    threading.Thread(target=_warmup_cpu, daemon=True).start()
    threading.Thread(target=_prefetch_gpu_model, daemon=True).start()
    return {"ok": True, "started": True, "device": DEVICE, "compute_type": COMPUTE_TYPE}

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
    # warmup opportuniste (ne bloque pas)
    if not WARMED_CPU:
        threading.Thread(target=_warmup_cpu, daemon=True).start()
    if (engine.lower() in ("auto", "gpu")) and (DEVICE == "cuda") and (not WARMED_GPU):
        threading.Thread(target=_prefetch_gpu_model, daemon=True).start()

    wav_path = _extract_audio_16k_mono(in_path)
    try:
        try:
            if engine.lower() == "cpu" or DEVICE != "cuda":
                asr_cpu = _ensure_cpu_model()
                segments_iter, info = asr_cpu.transcribe(
                    wav_path, language=language, vad_filter=VAD_FILTER, beam_size=BEAM_SIZE
                )
                segments_list = [{"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
                                 for s in segments_iter]
                lang = language or getattr(info, "language", "en")
                dbg = {"engine": "cpu", "forced": engine.lower() == "cpu"}
            elif engine.lower() in ("auto", "gpu"):
                segments_list, lang, dbg = _asr_transcribe_safe_gpu_then_cpu(wav_path, language)
            else:
                return JSONResponse(status_code=400, content={"ok": False, "error": "engine must be auto|cpu|gpu"})
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "ok": False, "stage": "asr", "error": str(e), "trace": traceback.format_exc()
            })

        try:
            align_model, metadata = _get_align_model(lang)
            wx = _wx()
            aligned = wx.align(
                segments_list, align_model, metadata, wav_path, ALIGN_DEVICE, return_char_alignments=False
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "ok": False, "stage": "whisperx.align", "error": str(e), "trace": traceback.format_exc(), "dbg": dbg
            })

        if DIARIZATION:
            try:
                diar = _ensure_diarization()
                if diar is not None:
                    wx = _wx()
                    diar_segments = diar(wav_path)
                    aligned = wx.assign_word_speakers(diar_segments, aligned)
            except Exception as e:
                return JSONResponse(status_code=500, content={
                    "ok": False, "stage": "diarization", "error": str(e), "trace": traceback.format_exc(), "dbg": dbg
                })

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
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8011")), reload=False)
