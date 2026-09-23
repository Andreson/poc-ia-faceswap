# RunPod Serverless GPU worker image. See REQ-OPS-017.
#
# Build context must be workers/ (this Dockerfile only needs workers/gpu_worker/):
#   docker build -f workers/docker/gpu_worker.Dockerfile -t <registry>/<repo>:<tag> workers/
#   docker push <registry>/<repo>:<tag>
# Then point a RunPod Serverless Endpoint at the pushed image (REQ-OPS-019).
# Prefer workers/docker/build_and_push.sh, which sets this up for you.
#
# Note: REQ-ARCH-007 originally placed Dockerfiles under a top-level docker/
# directory; this worker's image now lives under workers/docker/ instead.

ARG CUDA_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ---- Builder stage: resolve Python deps into a venv --------------------
FROM ${CUDA_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY gpu_worker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage: slim image with only what's needed to run ----------
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/runpod-volume/models

WORKDIR /app
COPY gpu_worker/ .

# REQ-AI-012: runpod.serverless.start() runs in handler.py itself.
CMD ["python3", "-u", "handler.py"]
