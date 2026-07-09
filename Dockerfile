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
    COMFY_TIMEOUT_S=3600 \
    HF_HOME=/runpod-volume/.cache/huggingface \
    HF_HUB_CACHE=/runpod-volume/.cache/huggingface/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      git \
      git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/venv/bin/python -m pip install --no-cache-dir "huggingface_hub[cli]"

# Wan2.2 repo를 고정 커밋으로 clone한다. unpinned main은 upstream 모듈 추가
# (예: wan/animate.py의 peft 의존)로 rebuild 시 조용히 깨질 수 있어 SHA로 고정.
ARG WAN22_REF=42bf4cfaa384bc21833865abc2f9e6c0e67233dc
RUN git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2 \
    && cd /opt/Wan2.2 \
    && git checkout "${WAN22_REF}" \
    && grep -v '^flash_attn' requirements.txt > /tmp/wan22-requirements-no-flash-attn.txt \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
      -r /tmp/wan22-requirements-no-flash-attn.txt \
      decord \
      librosa \
      peft \
      einops \
      ninja packaging \
    # flash-attn은 설치하지 않으므로(빌드 시간·GPU 아키텍처 호환 문제)
    # model.py가 직접 호출하는 flash_attention을 SDPA fallback이 있는
    # attention() wrapper로 우회시킨다. 미패치 시 attention.py:112의
    # `assert FLASH_ATTN_2_AVAILABLE`로 모든 영상 생성이 실패한다.
    && sed -i 's/^from \.attention import flash_attention$/from .attention import attention as flash_attention/' \
      /opt/Wan2.2/wan/modules/model.py \
    && grep -q "import attention as flash_attention" /opt/Wan2.2/wan/modules/model.py

# CPU-safe import smoke: `import wan`이 module-level에서 요구하는 서드파티
# 의존성이 전부 설치됐는지 빌드 단계에서 검증한다 (GPU 불필요).
# 목록은 위 WAN22_REF 커밋의 wan/ 패키지 import 클로저에서 도출했으며,
# WAN22_REF를 올릴 때 함께 재도출해야 한다. (직전 장애: wan/animate.py의
# peft import 누락이 런타임에서야 발견됨 — decord·librosa도 같은 사례)
RUN /opt/venv/bin/python -c "import PIL, cv2, decord, diffusers, easydict, einops, ftfy, imageio, librosa, numpy, peft, regex, safetensors, scipy, torch, torchaudio, torchvision, tqdm, transformers; print('wan deps import smoke OK')"

COPY handler.py /handler.py
COPY scripts/download_wan22_ti2v_5b.sh /usr/local/bin/download_wan22_ti2v_5b.sh
RUN chmod +x /usr/local/bin/download_wan22_ti2v_5b.sh

# The base worker-comfyui image starts ComfyUI and then runs /handler.py.
