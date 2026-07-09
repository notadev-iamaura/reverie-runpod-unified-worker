ARG BASE_IMAGE=runpod/worker-comfyui:5.8.6-base-cuda12.8.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIDEO_ENGINE=wan22_cli \
    WAN22_REPO_DIR=/opt/Wan2.2 \
    WAN22_MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B \
    WAN22_NATIVE_FPS=24 \
    WAN22_SAMPLE_STEPS=24 \
    WAN22_AUTO_DOWNLOAD=false \
    COMFY_TIMEOUT_S=1800

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      git \
      git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/venv/bin/python -m pip install --no-cache-dir "huggingface_hub[cli]"

ARG WAN22_REF=main
RUN git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2 \
    && cd /opt/Wan2.2 \
    && git checkout "${WAN22_REF}" \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY handler.py /handler.py
COPY scripts/download_wan22_ti2v_5b.sh /usr/local/bin/download_wan22_ti2v_5b.sh
RUN chmod +x /usr/local/bin/download_wan22_ti2v_5b.sh

# The base worker-comfyui image starts ComfyUI and then runs /handler.py.
