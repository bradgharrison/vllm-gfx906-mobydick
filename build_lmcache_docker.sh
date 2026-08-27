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
#   - mobydick   (aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0,
#                pinned to the official release — the fork's releases are all
#                `rc`-tagged, no non-rc "stable" exists)
#                -> docker/Dockerfile.mobydick-lmcache      (imports lmcache.c_ops)
#   - unverbraucht (unverbraucht/vllm-gfx906:<ver>-rocm-<ver>)
#     e.g. :0.26.0-rocm-7.2.1 or :0.28.0rc2-rocm-7.14
#                -> docker/Dockerfile.unverbraucht-lmcache
#                   (imports lmcache.cuda_ops / lmcache.lmcache_native)
#   The output tag is derived from the base tag (the -rocm-*/-cu* suffix is
#   stripped), so 0.28.0rc2-rocm-7.14 -> vllm-gfx906-lmcache:0.28.0rc2.
# The script picks the matching Dockerfile from BASE_IMAGE (override with
# DOCKERFILE).
#
# Usage:
#   ./build_lmcache_docker.sh                    # default: pinned mobydick base
#   # or target an unverbraucht ROCm base (tag auto-derived from the base tag):
#   BASE_IMAGE=unverbraucht/vllm-gfx906:0.28.0rc2-rocm-7.14 ./build_lmcache_docker.sh
#   #   -> vllm-gfx906-lmcache:0.28.0rc2
#   BASE_IMAGE=unverbraucht/vllm-gfx906:0.26.0-rocm-7.2.1 ./build_lmcache_docker.sh
#   #   -> vllm-gfx906-lmcache:0.26.0
#
# Env overrides:
#   IMAGE_NAME  (default: vllm-gfx906-lmcache — local tag only, not pushed)
#   BASE_IMAGE  (default: aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0;
#                or the unverbraucht 0.26.0 image above)
#   TAG         (default: auto — "mobydick" or "0.26.0" to match BASE_IMAGE)
#   LMCACHE_REF (default: dev — a branch/tag/commit of LMCache/LMCache)
#   DOCKERFILE  (default: auto — docker/Dockerfile.mobydick-lmcache or
#                docker/Dockerfile.unverbraucht-lmcache based on BASE_IMAGE)

IMAGE_NAME="${IMAGE_NAME:-vllm-gfx906-lmcache}"
BASE_IMAGE="${BASE_IMAGE:-aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0}"
LMCACHE_REF="${LMCACHE_REF:-dev}"
TAG="${TAG:-}"                 # auto-detected below if empty
DOCKERFILE="${DOCKERFILE:-}"   # auto-detected below if empty

# Pick the matching Dockerfile + tag from the base image (unless overridden).
# Tag mirrors the base version so the two builds pair up: :mobydick and the
# unverbraucht version (e.g. 0.26.0 -> :0.26.0, 0.28.0rc2-rocm-7.14 -> :0.28.0rc2).
if [ -z "${DOCKERFILE:-}" ]; then
    case "${BASE_IMAGE}" in
        *unverbraucht*)
            DOCKERFILE="docker/Dockerfile.unverbraucht-lmcache"
            # Derive the output tag from the base tag: 0.28.0rc2-rocm-7.14 -> 0.28.0rc2
            # (strip the trailing -rocm-<x> / -cu< x> toolchain suffix, if present).
            if [ -z "${TAG:-}" ]; then
                BASE_TAG="${BASE_IMAGE##*:}"            # 0.28.0rc2-rocm-7.14
                BASE_TAG="${BASE_TAG%%-rocm-*}"          # 0.28.0rc2
                BASE_TAG="${BASE_TAG%%-cu*}"              # 0.28.0rc2
                TAG="${BASE_TAG:-unverbraucht}"
            fi
            ;;
        *)
            DOCKERFILE="docker/Dockerfile.mobydick-lmcache"
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