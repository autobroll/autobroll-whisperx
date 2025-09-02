# Image CUDA 12.1 + cuDNN 8 (runtime) pour Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python + ffmpeg (indispensable pour l'extraction audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- D'abord Torch/Torchaudio en cu121 (compatibles cuDNN 8 de cette image) ---
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.0+cu121 torchaudio==2.1.0+cu121

# --- Puis le reste des deps ---
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Code ---
COPY . .

# Variables par défaut (overridable via RunPod)
ENV PORT=8011 \
    API_KEY=autobroll_secret_1 \
    WHISPERX_MODEL=large-v2 \
    WHISPERX_DEVICE=cuda \
    WHISPERX_COMPUTE_TYPE=float16 \
    WHISPERX_DIARIZATION=false

EXPOSE 8011

# Lancement API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", "--proxy-headers", "--log-level", "info"]
