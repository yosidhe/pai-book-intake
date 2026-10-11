FROM python:3.11-slim

# WSL2 の Docker には NVIDIA の EGL が来ないため、描画は mesa（CPU）で行う。物理計算は CUDA（GPU）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglib2.0-0 libosmesa6 libsm6 libxext6 libxrender1 \
        ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

# ホストのドライバ 561 系は CUDA 12.6 まで → cu126 の wheel を使う
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
