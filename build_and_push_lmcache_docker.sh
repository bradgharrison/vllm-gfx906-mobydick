#!/bin/bash
set -e

# Build (and optionally push) the vLLM-gfx906 + LMCache combined image.
# Modeled on build_and_push_docker.sh; reuses the mobydick base image
# rather than rebuilding vLLM from source.
#
# Usage:
#   ./build_and_push_lmcache_docker.sh                       # build only
#   ./build_and_push_lmcache_docker.sh push <dockerhub-user> # build + push
#
# Env overrides:
#   IMAGE_NAME  (default: aiinfos/vllm-gfx906-lmcache)
#   BASE_IMAGE  (default: aiinfos/vllm-gfx906-mobydick:latest)
#   LMCACHE_REF (default: dev — a branch/tag/commit of LMCache/LMCache)

IMAGE_NAME="${IMAGE_NAME:-aiinfos/vllm-gfx906-lmcache}"
BASE_IMAGE="${BASE_IMAGE:-aiinfos/vllm-gfx906-mobydick:latest}"
LMCACHE_REF="${LMCACHE_REF:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(mktemp -d /tmp/lmcache-build.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Using base image: ${BASE_IMAGE}"
echo "Using LMCache ref: ${LMCACHE_REF}"

# Build context = repo root (for docker/Dockerfile.lmcache) + LMCache clone
cd "${SCRIPT_DIR}/.."
if [ ! -d LMCache ]; then
    echo "Cloning LMCache at ref ${LMCACHE_REF}..."
    git clone --depth 1 --branch "${LMCACHE_REF}" \
        https://github.com/LMCache/LMCache.git LMCache \
        || git clone https://github.com/LMCache/LMCache.git LMCache
fi

echo "Building ${IMAGE_NAME}:latest ..."
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_NAME}:latest" \
    -f docker/Dockerfile.lmcache .

if [ "$1" = "push" ]; then
    DOCKER_USER="${2:?usage: $0 push <dockerhub-user>}"
    echo "Pushing to Docker Hub as ${DOCKER_USER}/vllm-gfx906-lmcache:latest ..."
    docker tag "${IMAGE_NAME}:latest" "${DOCKER_USER}/vllm-gfx906-lmcache:latest"
    docker login -u "${DOCKER_USER}"
    docker push "${DOCKER_USER}/vllm-gfx906-lmcache:latest"
else
    echo "Build complete. Verify with:"
    echo "  docker run --rm ${IMAGE_NAME}:latest python3 -c 'import lmcache, lmcache.c_ops, vllm; print(\"OK\")'"
fi
