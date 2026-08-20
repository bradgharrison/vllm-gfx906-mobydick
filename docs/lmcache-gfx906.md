# LMCache Integration for gfx906 (AMD MI50/MI60)

Serving [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (and the wider
Qwen3.5/Qwen3.6 hybrid family) on MI50/MI60 (gfx906) with
[LMCache](https://github.com/LMCache/LMCache) KV-cache offloading — verified
end-to-end: storage, retrieval, two-tier (RAM + NVMe) cascade, and multi-GPU
tensor parallelism.

## Why this exists

Qwen3.5/3.6 interleave Mamba/Gated-DeltaNet (GDN) linear-attention layers
with full-attention layers (`Qwen3_5ForConditionalGeneration`). LMCache
supports this hybrid layout via `mamba_cache_mode="align"`, but no prebuilt
image ships HIP-compiled LMCache against this fork's ROCm/torch stack.
`docker/Dockerfile.lmcache` builds that image; this page documents how to
run it.

## Image

One image, two roles (keep both from the same build — the ZMQ wire protocol
and CUDA-IPC wrappers are version-matched):

| Role | Command |
|---|---|
| vLLM engine + LMCacheMPConnector | `vllm serve ...` |
| LMCache MP-mode cache server | `lmcache server ...` |

Build (from the repo root — the script clones LMCache into the context):

```bash
./build_and_push_lmcache_docker.sh
```

The image layers on top of `aiinfos/vllm-gfx906-mobydick:latest`:
- LMCache built from source with `BUILD_WITH_HIP=1 CXX=hipcc`
  (HIP c_ops ABI-matched to the base torch)
- `cupy-rocm-7-0` (GPU stream management)
- `cufile-python` removed (NVIDIA-only; ROCm uses hipFile at runtime)
- `grpcio` re-pinned to the fork's requirement after LMCache's deps bump it

## The three magic numbers (Qwen3.6-27B)

vLLM logs `Setting attention block size to 784 tokens` at startup — the
model's unified block size **N**:

| Setting | Where | Value | Why |
|---|---|---|---|
| `--chunk-size` | lmcache server | **784** (= N) | chunk must equal block size |
| `--separate-object-groups` | lmcache server | flag | required for hybrid models |
| `--max-num-batched-tokens` | vllm serve | **1567** (= 2N−1) | finest Mamba snapshot granularity + decode co-scheduling |
| `--mamba-cache-mode align` | vllm serve | flag | GDN has no `all` mode |
| `--enable-prefix-caching` | vllm serve | flag | prerequisite |

Values ≥ 2N for `--max-num-batched-tokens` are legal with
`--separate-object-groups` on the server but snapshot the Mamba state at
coarser step boundaries; 2N−1 is the validated sweet spot. Setting it
exactly to N serializes prefill/decode once any request is decoding.

## Launch

### 1. LMCache server (must be running before vLLM connects)

```bash
mkdir -p /path/to/lmcache-disk

docker run -d --name lmcache-server --restart unless-stopped \
  --network host \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --env HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v /path/to/lmcache-disk:/lmcache-disk \
  <image> \
  lmcache server \
    --host 0.0.0.0 --port 5555 --http-port 8082 \
    --chunk-size 784 \
    --separate-object-groups \
    --l1-size-gb 60 \
    --eviction-policy LRU \
    --l2-adapter '{"type":"fs","base_path":"/lmcache-disk"}'
```

- ZMQ port 5555: engine → cache traffic. HTTP port 8082: management /
  metrics (`/cache/objects`, `/cache/clear`, `/status`, `/metrics`).
  Override the HTTP port — the 8080 default usually collides.
- The server must see **every GPU the vLLM workers use**
  (`HIP_VISIBLE_DEVICES`): it opens CUDA-IPC handles into the workers' KV
  tensors.
- The fs L2 adapter's key is `base_path` (not `path`).
- The fs adapter has no size cap — bound it by partition/free space. L2
  files are content-addressed and survive server restarts.

### 2. vLLM

```bash
docker run -d --name vllm-qwen \
  --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --env HIP_VISIBLE_DEVICES=0,1,2,3 \
  --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -v /path/to/models:/models \
  <image> \
  vllm serve /models/Qwen3.6-27B-INT8-AutoRound \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --max-num-batched-tokens 1567 \
    --enable-chunked-prefill \
    --max-model-seqs 16 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8443 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555}}'
```

If the cache server runs on another host, point `lmcache.mp.host` at its
LAN address instead of `tcp://127.0.0.1`.

## Verification

1. Startup sanity (vLLM logs): `Setting attention block size to 784 tokens`,
   `KV cache group edits applied: {'mamba-page-view': N}`,
   `Using external LMCacheMPConnector from lmcache.integration.vllm...`
2. Send a prompt longer than 784 tokens twice. lmcache-server logs:
   `Stored 784 tokens ...` then `Retrieved NNNN tokens ...`; the second
   request's TTFT drops 8–20×.
3. Tier isolation: `curl -X POST :8082/cache/clear -d '{"tier":"l1"}'`
   forces the next resend through NVMe (logs show `(0 L1, N L2)`).

## Measured performance (4× MI50, TP=4 / 2× MI50, TP=2 — 4k-token prompts)
(Dual x99, DDR3 PCI 3.0 4x4x4x4 PLX Active PLX Switch P2P via RCCL)

| Tier | TTFT | Effective PP |
|---|---|---|
| Cold (no cache) | 12.5 s | ~320 tok/s |
| VRAM prefix cache / L1 RAM hit | 0.6 s | ~6,900 tok/s |
| L2 NVMe hit (L1 cleared) | 0.8 s | ~5,250 tok/s |

With my setup, cold prefill is compute-bound (~310–320 tok/s at these sizes); any cache
hit is 16–22×. L2 NVMe adds only ~0.2 s over L1 RAM — the two-tier setup
effectively extends near-RAM-speed KV cache to the size of your disk.

## Caveats

- Mamba prefix caching in `align` mode is experimental upstream (vLLM warns
  at startup). Generation is not bit-exact between cached and fresh runs —
  expect score-level equivalence.
- Do NOT set `HSA_OVERRIDE_GFX_VERSION` on containers sharing a Triton
  cache with a different environment — it invalidates cached Triton modules
  (`cannot get address for 'hipGetErrorString'`).
- `VLLM_SLEEP_WHEN_IDLE=1` + LMCache: idle-sleeping stops heartbeats; the
  server reaps worker contexts after 120 s and they re-register on wake.
  Raise `--worker-reap-timeout-seconds` if the first request after a long
  idle misses the cache.

## Acknowledgements

- [ai-infos/vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick) — the gfx906 vLLM fork and base image
- [LMCache/LMCache](https://github.com/LMCache/LMCache) — the cache engine and the [Qwen3.5/3.6 recipe](https://docs.lmcache.ai/recipes/qwen3_5.html) this validates
