#!/bin/bash
set -e

# Build the vLLM-gfx906 + LMCache combined image (local use only).
#
# This image is NOT published anywhere — build it locally on your own
# machine. There is intentionally no push step: the base image comes from
# the aiinfos/vllm-gfx906-mobydick image, and you layer LMCache on top.
#
# Modeled on build_and_push_docker.sh; reuses a base vLLM-gfx906 image
# rather than rebuilding vLLM from source.
#
# Two base-image generations are supported:
#   - mobydick   (aiinfos/vllm-gfx906-mobydick:latest)
#                -> docker/Dockerfile.mobydick-lmcache      (imports lmcache.c_ops)
#   - unverbraucht (unverbraucht/vllm-gfx906:0.26.0-rocm-7.2.1, vLLM 0.26.0)
#                -> docker/Dockerfile.unverbraucht-lmcache
#                   (imports lmcache.cuda_ops / lmcache.lmcache_native)
# The script picks the matching Dockerfile from BASE_IMAGE (override with
# DOCKERFILE).
#
# Usage:
#   ./build_lmcache_docker.sh
#   # or target the 0.26.0 base (tags vllm-gfx906-lmcache:0.26.0):
#   BASE_IMAGE=unverbraucht/vllm-gfx906:0.26.0-rocm-7.2.1 ./build_lmcache_docker.sh
#
# Env overrides:
#   IMAGE_NAME  (default: vllm-gfx906-lmcache — local tag only, not pushed)
#   BASE_IMAGE  (default: aiinfos/vllm-gfx906-mobydick:latest; or the
#                unverbraucht 0.26.0 image above)
#   TAG         (default: auto — "mobydick" or "0.26.0" to match BASE_IMAGE)
#   LMCACHE_REF (default: dev — a branch/tag/commit of LMCache/LMCache)
#   DOCKERFILE  (default: auto — docker/Dockerfile.mobydick-lmcache or
#                docker/Dockerfile.unverbraucht-lmcache based on BASE_IMAGE)

IMAGE_NAME="${IMAGE_NAME:-vllm-gfx906-lmcache}"
BASE_IMAGE="${BASE_IMAGE:-aiinfos/vllm-gfx906-mobydick:latest}"
LMCACHE_REF="${LMCACHE_REF:-dev}"
TAG="${TAG:-}"                 # auto-detected below if empty
DOCKERFILE="${DOCKERFILE:-}"   # auto-detected below if empty

# Pick the matching Dockerfile + tag from the base image (unless overridden).
# Tag mirrors the base so the two builds pair up: :mobydick and :0.26.0.
if [ -z "${DOCKERFILE:-}" ]; then
    case "${BASE_IMAGE}" in
        *unverbraucht*) DOCKERFILE="docker/Dockerfile.unverbraucht-lmcache"
                        TAG="${TAG:-0.26.0}" ;;
        *)              DOCKERFILE="docker/Dockerfile.mobydick-lmcache"
                        TAG="${TAG:-mobydick}" ;;
    esac
fi
# 0.26.0-era wheel renamed lmcache.c_ops -> lmcache.cuda_ops/lmcache_native
if [ "${DOCKERFILE}" = "docker/Dockerfile.unverbraucht-lmcache" ]; then
    IMPORT="import lmcache, lmcache.cuda_ops, lmcache.lmcache_native, vllm"
    IMPORT_LABEL="cuda_ops/lmcache_native"
else
    IMPORT="import lmcache, lmcache.c_ops, vllm"
    IMPORT_LABEL="c_ops"
fi
# Fall back to a per-Dockerfile tag if TAG wasn't set (covers DOCKERFILE= override).
if [ -z "${TAG:-}" ]; then
    case "${DOCKERFILE}" in
        *unverbraucht-lmcache) TAG="0.26.0" ;;
        *) TAG="mobydick" ;;
    esac
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(mktemp -d /tmp/lmcache-build.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Using base image: ${BASE_IMAGE}"
echo "Using LMCache ref: ${LMCACHE_REF}"
echo "Using Dockerfile:  ${DOCKERFILE}"
echo "Using image name:  ${IMAGE_NAME}:${TAG}"

# Build context = repo root (for docker/Dockerfile.mobydick-lmcache) + LMCache clone
cd "${SCRIPT_DIR}/.."
if [ ! -d LMCache ]; then
    echo "Cloning LMCache at ref ${LMCACHE_REF}..."
    git clone --depth 1 --branch "${LMCACHE_REF}" \
        https://github.com/LMCache/LMCache.git LMCache \
        || git clone https://github.com/LMCache/LMCache.git LMCache
fi

echo "Building ${IMAGE_NAME}:${TAG} ..."
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${DOCKERFILE}" .

# Keep a :latest alias for the common single-build case.
docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest"

echo "Build complete. Verify with:"
echo "  docker run --rm ${IMAGE_NAME}:${TAG} python3 -c '${IMPORT}; print(\"OK\")'   # ${IMPORT_LABEL}"