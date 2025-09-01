FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python + ffmpeg (indispensable pour l'extraction audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les deps Python (versions figées dans requirements.txt)
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API (main.py etc.)
COPY . .

# Variables par défaut (peuvent être override dans RunPod Serverless)
ENV PORT=8011 \
    API_KEY=autobroll_secret_1 \
    WHISPERX_MODEL=large-v2 \
    WHISPERX_DEVICE=cuda \
    WHISPERX_COMPUTE_TYPE=float16 \
    WHISPERX_DIARIZATION=false

EXPOSE 8011

# En serverless (pas de proxy Jupyter), on lance directement uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", "--proxy-headers", "--log-level", "info"]
