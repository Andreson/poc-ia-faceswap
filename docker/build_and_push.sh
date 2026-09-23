#!/usr/bin/env bash
# Build and push the gpu_worker RunPod Serverless image to Docker Hub.
# See REQ-OPS-017 (.harness/04-implementation_roadmap_harness.md) and
# workers/docker/gpu_worker.Dockerfile. Build context is workers/ (this
# script's parent directory), not the repo root.
#
# Usage (run from anywhere; resolves paths itself):
#   ./workers/docker/build_and_push.sh [tag]
#
# Requires `docker login` to have been run already for the Docker Hub
# account below (or set DOCKERHUB_USER to override).
#
# Always builds for linux/amd64: RunPod GPU hosts are x86_64, and
# onnxruntime-gpu (requirements.txt) publishes no linux/aarch64 wheels at
# all, so building on an Apple Silicon / arm64 machine without pinning the
# platform fails pip's dependency resolution. Building for amd64 on an
# arm64 host runs under QEMU emulation and is noticeably slower.

set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-andreson09thiago}"
IMAGE_NAME="${IMAGE_NAME:-faceswap}"
TAG="${1:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "Building ${IMAGE} (${PLATFORM}) from ${WORKERS_DIR} (docker/gpu_worker.Dockerfile) ..."
docker build \
    --platform "${PLATFORM}" \
    -f "${SCRIPT_DIR}/gpu_worker.Dockerfile" \
    -t "${IMAGE}" \
    "${WORKERS_DIR}"

echo "Pushing ${IMAGE} ..."
docker push "${IMAGE}"

echo "Done: ${IMAGE}"
echo "Point the RunPod Serverless Endpoint's container image at this tag (REQ-OPS-019)."
