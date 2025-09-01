# Image CUDA valide (avec cuDNN 8) pour Ubuntu 22.04
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

# Installer les deps Python (requirements.txt doit contenir la ligne
# --extra-index-url https://download.pytorch.org/whl/cu121
# pour récupérer les roues GPU de PyTorch)
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copier le code de l'API
COPY . .

# Variables par défaut (écrasables via les variables d'env du host)
ENV PORT=8011 \
    API_KEY=autobroll_secret_1 \
    WHISPERX_MODEL=large-v2 \
    WHISPERX_DEVICE=cuda \
    WHISPERX_COMPUTE_TYPE=float16 \
    WHISPERX_DIARIZATION=false

EXPOSE 8011

# Lance l'API (serverless / sans Jupyter)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", "--proxy-headers", "--log-level", "info"]
