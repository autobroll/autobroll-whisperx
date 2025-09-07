# === Base PyTorch alignée CUDA 12.1 + cuDNN 9 ===
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Outils + ffmpeg (audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv ffmpeg git git-lfs ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances applis (torch/torchaudio déjà fournis par l'image)
COPY requirements.txt constraints.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Code
COPY . .

# Caches LOCAUX (pas de bucket) — on bake dans /root/.cache
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache \
    TORCH_HOME=/root/.cache/torch \
    NUMBA_CACHE_DIR=/root/.cache/numba \
    CUDA_CACHE_PATH=/root/.cache/cuda \
    TOKENIZERS_PARALLELISM=false \
    CT2_FORCE_CPU_ISA=AVX2 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ---------- Prefetch des modèles pour éviter tout download en prod ----------

# 1) Poids CTranslate2 de faster-whisper (large-v3) – téléchargés en CPU au build
#    => À l'exécution on chargera en CUDA sans re-télécharger.
RUN python - <<'PY'
import os
from faster_whisper import WhisperModel
os.makedirs(os.environ.get("HF_HOME","/root/.cache/huggingface"), exist_ok=True)
# télécharge en CPU (int8) juste pour remplir le cache
WhisperModel("large-v3", device="cpu", compute_type="int8")
print("Prefetch faster-whisper large-v3 OK")
PY

# 2) Checkpoint torchaudio FR (360 Mo) – placé dans TORCH_HOME
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L \
  -o /root/.cache/torch/hub/checkpoints/wav2vec2_voxpopuli_base_10k_asr_fr.pt \
     https://download.pytorch.org/torchaudio/models/wav2vec2_voxpopuli_base_10k_asr_fr.pt && \
    echo "Prefetch torchaudio wav2vec2 FR OK"

# 3) Prefetch des modèles d’alignement WhisperX (en + fr)
#    On les télécharge en CPU au build pour remplir HF_HOME/TRANSFORMERS_CACHE.
RUN python - <<'PY'
import os
import whisperx
os.makedirs(os.environ.get("HF_HOME","/root/.cache/huggingface"), exist_ok=True)
for lang in ["en", "fr"]:
    try:
        m, meta = whisperx.load_align_model(language_code=lang, device="cpu")
        print("Prefetch whisperx aligner OK:", lang, meta)
    except Exception as e:
        print("Prefetch whisperx aligner FAILED for", lang, "->", e)
PY

# Paramètres par défaut (surclassables dans Cloud Run)
ENV PORT=8011 \
    API_KEY=autobroll_secret_1 \
    WHISPERX_MODEL=large-v3 \
    WHISPERX_DEVICE=cuda \
    WHISPERX_ALIGN_DEVICE=cuda \
    WHISPERX_COMPUTE_TYPE=float16 \
    WHISPERX_DIARIZATION=false \
    WHISPERX_BEAM_SIZE=1 \
    WHISPERX_VAD_FILTER=true \
    WHISPERX_DEVICE_INDEX=0 \
    WHISPERX_ASR_GPU_TIMEOUT=240

EXPOSE 8011
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --log-level info"]
