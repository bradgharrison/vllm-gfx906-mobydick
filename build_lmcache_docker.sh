#!/bin/bash
set -e

# Build the vLLM-gfx906 + LMCache combined image (local use only).
#
# This image is NOT published anywhere — build it locally on your own
# machine. There is intentionally no push step: the base image comes from
# the aiinfos/vllm-gfx906-mobydick image, and you layer LMCache on top.
#
# Modeled on build_and_push_docker.sh; reuses the mobydick base image
# rather than rebuilding vLLM from source.
#
# Usage:
#   ./build_lmcache_docker.sh
#
# Env overrides:
#   IMAGE_NAME  (default: aiinfos/vllm-gfx906-lmcache — local tag only, not pushed)
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

echo "Build complete. Verify with:"
echo "  docker run --rm ${IMAGE_NAME}:latest python3 -c 'import lmcache, lmcache.c_ops, vllm; print(\"OK\")'"