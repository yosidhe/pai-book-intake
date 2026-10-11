FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglib2.0-0 libosmesa6 libsm6 libxext6 libxrender1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
