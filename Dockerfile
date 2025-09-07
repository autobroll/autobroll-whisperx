# === PyTorch + CUDA 12.1 + cuDNN 9 (runtime) ===
# Aligne Torch / CUDA / cuDNN pour éviter tout fallback CPU
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Outils & ffmpeg (pour l’audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv ffmpeg git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⚠️ IMPORTANT : on ne réinstalle PAS torch/torchaudio ici (déjà fournis par l'image)
# Installe seulement tes deps applicatives
COPY requirements.txt constraints.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Code
COPY . .

# Caches (pointeront vers le volume GCS si tu le montes sur /cache)
ENV HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    XDG_CACHE_HOME=/cache \
    TORCH_HOME=/cache/torch \
    NUMBA_CACHE_DIR=/cache/numba \
    CUDA_CACHE_PATH=/cache/cuda \
    TOKENIZERS_PARALLELISM=false \
    CT2_FORCE_CPU_ISA=AVX2

# Paramètres par défaut (tu peux override dans Cloud Run)
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
