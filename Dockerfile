# Image CUDA 12.1 + cuDNN 8 (runtime) pour Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python + ffmpeg (indispensable pour l'extraction audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Torch/Torchaudio en cu121 (compatibles avec cette image) ---
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.0+cu121 torchaudio==2.1.0+cu121

# --- Dépendances de l'app ---
COPY requirements.txt constraints.txt ./
RUN echo "===== requirements.txt =====" && cat requirements.txt && \
    echo "===== constraints.txt =====" && cat constraints.txt && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

# --- Code ---
COPY . .

# --- Cache HuggingFace stable (utile pour éviter les re-downloads) ---
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    # CTranslate2: force une ISA CPU correcte quand on build sans GPU (sans incidence à l'exécution GPU)
    CT2_FORCE_CPU_ISA=AVX2

# --- Variables par défaut (override via RunPod) ---
# Astuce: on met des valeurs "raisonnables" pour GPU; vous pouvez les surcharger à l'exécution.
ENV PORT=8011 \
    API_KEY=autobroll_secret_1 \
    WHISPERX_MODEL=medium \
    WHISPERX_DEVICE=cuda \
    WHISPERX_ALIGN_DEVICE=cpu \
    WHISPERX_COMPUTE_TYPE=int8_float16 \
    WHISPERX_DIARIZATION=false \
    WHISPERX_BEAM_SIZE=1 \
    WHISPERX_VAD_FILTER=true \
    WHISPERX_DEVICE_INDEX=0 \
    WHISPERX_ASR_GPU_TIMEOUT=180

# --- Prefetch du modèle Faster-Whisper pendant le build ---
# On télécharge les poids CT2 à l'étape build (en "cpu") pour les embarquer dans l'image.
# Au runtime, le même modèle sera chargé en "cuda" sans re-télécharger.
RUN python3 - <<'PY'
import os
from faster_whisper import WhisperModel
model = os.environ.get("WHISPERX_MODEL", "medium")
compute = os.environ.get("WHISPERX_COMPUTE_TYPE", "int8_float16")
# Téléchargement des poids CT2 dans HF cache (CPU suffit au build)
WhisperModel(model, device="cpu", compute_type="int8")
print("Prefetch done for:", model, "(cpu,int8)")
PY

EXPOSE 8011

# Lancement API (respecte $PORT)
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --log-level info"]
