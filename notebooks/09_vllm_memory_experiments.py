# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # vLLM Memory & Performance Experiments
#
# This notebook documents all vLLM benchmarking experiments for the Qwen2-VL nutrition detection model.
#
# **Goals:**
# 1. Understand vLLM memory consumption (why VRAM doesn't change during requests)
# 2. Compare vLLM vs HuggingFace Transformers performance
# 3. Find optimal concurrency levels (compute-bound vs KV-cache-bound)
# 4. Test different memory utilization settings
#
# **Hardware:** 8x RTX 3090 (24GB vRAM each)

# %% [markdown]
# ## 1. Setup: Import Reusable Functions
#
# We reuse functions from `scripts/benchmark_vllm.py` to keep the notebook concise.

# %%
import sys
sys.path.insert(0, '/home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/scripts')

# Import reusable functions from our benchmark script
from benchmark_vllm import (
    run_benchmark,           # Main benchmark function
    get_metrics_snapshot,    # Get vLLM /metrics data
    compute_metrics_delta,   # Compute metrics difference
    VLLM_HOST, VLLM_PORT,    # Server configuration
)

import requests
import time

# %%
# Quick health check
def check_server():
    """Check if vLLM server is running."""
    try:
        r = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

if check_server():
    print("✓ vLLM server is running")
    metrics = get_metrics_snapshot()
    print(f"  KV Cache Usage: {metrics.kv_cache_usage * 100:.2f}%")
    print(f"  Requests Running: {metrics.requests_running}")
else:
    print("✗ Server not running. Start with commands below.")

# %% [markdown]
# ## 2. Server Start Commands
#
# Run these in terminal (not in notebook):
#
# ```bash
# # Standard (0.9 memory, GPU 0, port 8000)
# CUDA_VISIBLE_DEVICES=0 vllm serve \
#   /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
#   --served-model-name qwen2vl-nutrition \
#   --dtype bfloat16 --trust-remote-code \
#   --max-model-len 4096 --limit-mm-per-prompt '{"image":1}' \
#   --gpu-memory-utilization 0.9 --port 8000
# ```

# %% [markdown]
# ## 3. Understanding vLLM Memory Pre-allocation
#
# **Key Insight:** vLLM pre-allocates GPU memory at startup. VRAM doesn't change during inference.
#
# ```
# Memory Layout (RTX 3090, 24GB):
# ┌─────────────────────────────────────────────────────────────┐
# │  Model Weights (~15.5 GB for Qwen2-VL-7B)                   │
# ├─────────────────────────────────────────────────────────────┤
# │  KV Cache Pool (pre-allocated, ~5-6 GB at 0.9 util)         │
# │  └── Reused across requests (not allocated/freed)           │
# ├─────────────────────────────────────────────────────────────┤
# │  Reserved free space (~2.5 GB at 0.9 util)                  │
# └─────────────────────────────────────────────────────────────┘
# ```
#
# ### Memory Utilization Breakdown
#
# | Setting | VRAM Budget | Model | Available for KV Cache | Result |
# |---------|-------------|-------|------------------------|--------|
# | 0.9 | 22.1 GB | 15.5 GB | ~6.6 GB (5009 blocks) | Works |
# | 0.8 | 19.7 GB | 15.5 GB | ~4.2 GB (1.91 GiB) | Works |
# | 0.7 | 17.2 GB | 15.5 GB | ~1.7 GB | **Fails** |
# | 0.5 | 12.3 GB | 15.5 GB | Negative | **Fails** |
#
# **Formula:** `Available KV Cache = (GPU Memory × util) - Model Weights - CUDA Overhead`

# %% [markdown]
# ## 4. Two Types of Concurrency Limits
#
# ### Compute-Bound vs KV-Cache-Bound
#
# | Limit Type | What It Means | Symptom |
# |------------|---------------|---------|
# | **Compute-Bound** | GPU can't process tokens fast enough | Latency increases, throughput plateaus |
# | **KV-Cache-Bound** | No more memory for new requests | Requests queue (`num_requests_waiting > 0`) |
#
# **Analogy (Restaurant):**
# - **KV cache** = Table capacity (how many guests can sit)
# - **Compute** = Kitchen speed (how fast food is prepared)
#
# Even with empty tables, the kitchen can only serve so many customers per hour.

# %% [markdown]
# ## 5. Experiment: Prefix Caching Impact

# %%
# Run this to compare same-image vs different-image performance
# Uncomment to execute:

# print("=== Same Image (Best-case caching) ===")
# results_same = run_benchmark(num_requests=20, concurrency=8, vary_images=False)

# print("\n=== Different Images (Realistic) ===")
# results_diff = run_benchmark(num_requests=20, concurrency=8, vary_images=True)

# %% [markdown]
# ### Prefix Caching Results
#
# | Metric | Same Image | Different Images | Impact |
# |--------|------------|------------------|--------|
# | **Throughput (c=8)** | 11.40 req/s | 3.17 req/s | **-72%** |
# | **TTFT** | 42 ms | 719 ms | **17x slower** |
# | **E2E** | 534 ms | 2,138 ms | **4x slower** |

# %% [markdown]
# ## 6. Experiment: Concurrency Sweep (Compute-Bound)

# %%
# Run concurrency sweep to find compute-bound limit
# Uncomment to execute:

# for c in [1, 2, 4, 8, 16, 32]:
#     print(f"\n=== Concurrency = {c} ===")
#     results = run_benchmark(num_requests=c*4, concurrency=c, vary_images=True)

# %% [markdown]
# ### Concurrency Results (Different Images, Compute-Bound)
#
# | Concurrency | Throughput | TTFT | E2E Avg | KV Cache % (snapshot) | Theoretical Peak |
# |-------------|------------|------|---------|----------------------|------------------|
# | 1 | 1.09 req/s | 454 ms | 907 ms | ~1% | ~1.7% |
# | 8 | 3.17 req/s | 719 ms | 2,138 ms | ~2% | ~13% |
# | 16 | 4.98 req/s | 1,103 ms | 3,008 ms | ~2.3% | ~26% |
# | 32 | 4.20 req/s | 3,213 ms | 7,035 ms | ~3% | ~53% |
#
# **Note:** Snapshot measurements are low because they capture moments between request completions.
# During active processing, KV cache reaches the theoretical peak values.
#
# **Finding:** Throughput peaks at c=16, then decreases. System is compute-bound (GPU saturates
# at c=16 even though KV cache is only at ~26% capacity), not KV-cache-bound.

# %% [markdown]
# ## 7. Why Higher Concurrency = Higher Latency?
#
# Continuous batching **shares** GPU resources, it doesn't **multiply** them:
#
# ```
# Concurrency=1:
# ┌─────────────────────────────────────┐
# │  GPU: 100% dedicated to Request 1   │
# │  TTFT: 454ms, E2E: 907ms            │
# └─────────────────────────────────────┘
#
# Concurrency=8:
# ┌─────────────────────────────────────┐
# │  GPU: Split across 8 requests       │
# │  Each request gets ~1/8 compute     │
# │  TTFT: 719ms, E2E: 2138ms           │
# └─────────────────────────────────────┘
# ```
#
# **The tradeoff:**
# - **c=1**: Low latency (907ms), low throughput (1.09 req/s)
# - **c=8**: Higher latency (2138ms), higher throughput (3.17 req/s)
#
# Continuous batching improves **throughput**, not **latency**.

# %% [markdown]
# ## 8. Experiment: Find KV-Cache-Bound Limit
#
# To hit the KV-cache limit, we need to either:
# 1. Use many more concurrent requests
# 2. Use longer sequences
#
# vLLM startup log shows: `Maximum concurrency for 4,096 tokens per request: 8.74x`
#
# This means theoretically ~8-9 requests can fit in KV cache simultaneously.
# But compute becomes the bottleneck before we hit this limit.

# %%
def find_kv_cache_limit(max_concurrency=128, step=16):
    """
    Push concurrency until KV cache saturates or requests start queuing.
    """
    print("Finding KV-cache-bound limit...")
    print(f"{'Concurrency':>12} {'Throughput':>12} {'KV Cache %':>12} {'Waiting':>10}")
    print("-" * 50)

    for c in range(step, max_concurrency + 1, step):
        try:
            results = run_benchmark(
                num_requests=c * 2,
                concurrency=c,
                vary_images=True,
                verbose=False
            )

            # Get current metrics
            metrics = get_metrics_snapshot()
            kv_pct = metrics.kv_cache_usage * 100
            waiting = metrics.requests_waiting
            throughput = results['summary']['throughput_rps']

            print(f"{c:>12} {throughput:>12.2f} {kv_pct:>12.1f} {waiting:>10}")

            # Stop if KV cache > 80% or requests queuing significantly
            if kv_pct > 80 or waiting > c // 2:
                print(f"\n→ KV-cache limit reached at concurrency ~{c}")
                break

        except Exception as e:
            print(f"{c:>12} ERROR: {e}")
            break

    return c

# Uncomment to run:
# kv_limit = find_kv_cache_limit()

# %% [markdown]
# ### KV-Cache Limit Calculation
#
# From vLLM metrics (`/metrics` endpoint):
# ```
# num_gpu_blocks: 5009
# block_size: 16 tokens
# Total KV cache capacity: 5009 × 16 = 80,144 tokens
# ```
#
# **Per-request token usage:**
# - Prompt tokens: ~1,300 (system prompt + image + user prompt)
# - Generation tokens: ~24
# - Total: ~1,324 tokens per request
#
# **Theoretical KV-cache limit:**
# ```
# Max concurrent requests = 80,144 tokens / 1,324 tokens = ~60 requests
# ```
#
# ### High Concurrency Test Results
#
# | Concurrency | Throughput | E2E Latency | KV Cache % | Status |
# |-------------|------------|-------------|------------|--------|
# | 32 | 2.30 req/s | 13,654 ms | ~0% | Compute-bound |
# | 48 | 7.77 req/s | 4,055 ms | ~0% | Compute-bound |
# | 64 | 2.54 req/s | 23,469 ms | ~0% | Severe compute-bound |
# | 80 | 2.37 req/s | 28,964 ms | ~0% | Severe compute-bound |
#
# **Key Finding:** KV cache shows 0% because it's measured AFTER requests complete.
# The real bottleneck is **GPU compute**, not KV cache memory.
#
# ### Summary: Two Limits
#
# | Limit Type | Concurrency | Symptom | Our Case |
# |------------|-------------|---------|----------|
# | **Compute-bound** | ~16 | Latency increases, throughput plateaus | **This is our limit** |
# | **KV-cache-bound** | ~60 | `num_requests_waiting > 0` | Never reached |
#
# **Why compute-bound at c=16?**
# - Vision encoder is expensive (~389ms prefill per image)
# - Attention computation scales with batch size
# - Memory bandwidth limits token processing speed

# %% [markdown]
# ## 9. Memory Utilization Impact

# %%
# Pre-computed results from memory utilization experiments
MEMORY_RESULTS = {
    0.9: {"vram_gb": 22.9, "kv_cache_gib": 5.0, "status": "Works"},
    0.8: {"vram_gb": 20.4, "kv_cache_gib": 1.91, "status": "Works"},
    0.7: {"vram_gb": None, "kv_cache_gib": -0.46, "status": "FAILED"},
}

print("Memory Utilization Results")
print("=" * 50)
print(f"{'Setting':>10} {'VRAM (GB)':>12} {'KV Cache':>12} {'Status':>10}")
print("-" * 50)
for setting, r in MEMORY_RESULTS.items():
    vram = f"{r['vram_gb']:.1f}" if r['vram_gb'] else "N/A"
    kv = f"{r['kv_cache_gib']:.2f} GiB"
    print(f"{setting:>10} {vram:>12} {kv:>12} {r['status']:>10}")

# %% [markdown]
# ### Why 0.7 Memory Utilization Failed
#
# ```
# Model weights:           15.5 GB
# Memory budget at 0.7:    17.2 GB (0.7 × 24.5 GB)
# Remaining for KV cache:   1.7 GB
# After CUDA overhead:     NEGATIVE → Server cannot start
# ```
#
# **Minimum viable setting: 0.8** (leaves 1.9 GiB for KV cache)

# %% [markdown]
# ## 10. vLLM vs HuggingFace Comparison
#
# | Metric | HF Transformers | vLLM (c=1) | vLLM (c=8) |
# |--------|-----------------|------------|------------|
# | **Throughput** | 0.66 req/s | 1.09 req/s | 3.17 req/s |
# | **Improvement** | baseline | **1.6x** | **4.8x** |
# | **Latency** | 1,189 ms | 907 ms | 2,138 ms |
#
# **Why vLLM is faster:**
# 1. Optimized CUDA kernels
# 2. Continuous batching (processes multiple requests simultaneously)
# 3. Prefix caching (reuses KV cache for repeated prompts)
#
# **Why HF can't match throughput:**
# VLMs process images sequentially (different sizes/aspect ratios).
# No native batching for different images.

# %% [markdown]
# ## 11. Summary & Recommendations
#
# ### Production Settings
#
# | Setting | Value | Reason |
# |---------|-------|--------|
# | `--gpu-memory-utilization` | 0.9 | Maximum KV cache |
# | Max Concurrency | 8-16 | Best throughput/latency tradeoff |
# | Expected Throughput | 3-5 req/s | With diverse images |
# | Expected Latency | 2-3 seconds | At c=8-16 |
#
# ### Resume-ready Claim
#
# > "Deployed Qwen2-VL with vLLM, achieving **4.8x throughput** (3.17 vs 0.66 req/s)
# > compared to HuggingFace Transformers. Demonstrated **15-17x TTFT improvement**
# > through prefix caching for repeated queries."

# %% [markdown]
# ## 12. Future Work: Quantization
#
# **Goal:** Test if FP8 quantization provides speed/memory benefits.
#
# **Challenge:** RTX 3090 (Ampere) has limited FP8 support. Native FP8 works best on H100 (Hopper).
#
# **Options to explore:**
# 1. FP8 quantization (if supported)
# 2. AWQ quantization (requires pre-quantized model)
# 3. GPTQ quantization (requires pre-quantized model)
#
# ```bash
# # FP8 (may not work on RTX 3090)
# vllm serve ... --quantization fp8
#
# # AWQ (needs quantized model)
# vllm serve /path/to/awq-quantized-model --quantization awq
# ```
#
# **What to measure:**
# - Quality: IoU scores on validation set
# - Speed: TTFT, throughput
# - Memory: VRAM reduction, KV cache increase
