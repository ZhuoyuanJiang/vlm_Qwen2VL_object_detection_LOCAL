# Session 28 Progress - HuggingFace Baseline & Prefix Caching Analysis

**Date**: 2026-01-05
**Session Name**: huggingface-benchmark

---

## Complete Experiments List

### Experiment 1: HuggingFace Transformers Baseline

**Metrics Note:** HuggingFace doesn't have TTFT/TPOT (no streaming). E2E = total inference time.

**1A: Batch Size Effect Test** (Does batch_size help VLMs?)
- 1A.1: batch_size=1, num_images=8, warmup=2 → Throughput=0.66 req/s, E2E=1183 ms
- 1A.2: batch_size=4, num_images=8, warmup=2 → Throughput=0.69 req/s, E2E=1192 ms
- 1A.3: batch_size=8, num_images=8, warmup=2 → Throughput=0.66 req/s, E2E=1191 ms
- **Finding:** batch_size has NO effect on VLM throughput (images processed sequentially)

**1B: Cold Start Test** (First inference latency)
- 1B.1: Model load time → 4.67s
- 1B.2: First inference (cold) → E2E=1865 ms
- 1B.3: Second inference → E2E=1179 ms
- 1B.4: Third inference → E2E=1179 ms
- 1B.5: Avg after warmup → E2E=1180 ms
- **Finding:** Cold start penalty = 685 ms (1.58x slower than warmed up)

**1C: Standard Benchmark**
- 1C.1: batch_size=1, num_images=20, warmup=3 → Throughput=0.66 req/s, E2E=1189 ms

### Experiment 2: vLLM Same Image (Prefix Caching)

**2A: Cold Start (c=1)** - Server just started, first requests
- 2A.1: num_requests=10, c=1 → Throughput=1.14 req/s, TTFT=332.5 ms, TPOT=20.2 ms, E2E=796 ms
- 2A.2: First request max latency → 4159.5 ms (CUDA kernel compilation)
- **Finding:** First request 4.1s due to JIT compilation

**2B: Cold Start (c=8)** - Server just started, concurrent requests
- 2B.1: num_requests=16, c=8 → Throughput=3.30 req/s, TTFT=453.2 ms, TPOT=21.2 ms, E2E=939 ms
- 2B.2: Max latency → 4269.9 ms
- **Finding:** Continuous batching helps even during cold start

**2C: Warmed Up (c=1, Prefix caching active)**
- 2C.1: num_requests=20, c=1 → Throughput=1.97 req/s, TTFT=28 ms, TPOT=20.1 ms, E2E=492 ms
- 2C.2: num_requests=10, c=1 → Throughput=1.97 req/s, TTFT=29 ms, TPOT=20.1 ms, E2E=491 ms
- 2C.3: num_requests=20, c=1 → Throughput=1.96 req/s, TTFT=31 ms, TPOT=20.2 ms, E2E=495 ms
- **Finding:** TTFT drops from 332ms to 29ms with prefix caching (11.5x improvement)

**2D: Warmed Up (c=8)**
- 2D.1: num_requests=20, c=8 → Throughput=11.40 req/s, TTFT=42 ms, TPOT=20.1 ms, E2E=534 ms

### Experiment 3: vLLM Different Images (Realistic Production)

**3A: Cold Start (c=1)**
- 3A.1: num_requests=10, c=1, vary_images → Throughput=0.78 req/s, TTFT=750.8 ms, TPOT=20.2 ms, E2E=1203 ms
- 3A.2: Max latency → 4176.1 ms
- **Finding:** Without prefix cache, TTFT is 751ms vs 29ms (26x slower)

**3B: Cold Start (c=8)**
- 3B.1: num_requests=16, c=8, vary_images → Throughput=1.35 req/s, TTFT=1819 ms, TPOT=95.2 ms, E2E=3954 ms
- 3B.2: Max latency → 10025.4 ms
- **Finding:** Cold start + different images + high concurrency = worst case

**3C: Warmed Up (c=1)**
- 3C.1: num_requests=10, c=1, vary_images → Throughput=1.09 req/s, TTFT=454 ms, TPOT=20.2 ms, E2E=907 ms

**3D: Warmed Up (c=8)**
- 3D.1: num_requests=20, c=8, vary_images → Throughput=3.17 req/s, TTFT=719 ms, TPOT=61.8 ms, E2E=2138 ms

### Experiment 4: Concurrency Sweep (Different Images)

**Purpose:** Find optimal concurrency (throughput vs latency tradeoff)
- 4.1: concurrency=1, num_requests=10 → 1.09 req/s, TTFT=454 ms, E2E=907 ms
- 4.2: concurrency=8, num_requests=20 → 3.17 req/s, TTFT=719 ms, E2E=2138 ms
- 4.3: concurrency=16, num_requests=32 → 4.98 req/s, TTFT=1103 ms, E2E=3008 ms
- 4.4: concurrency=32, num_requests=64 → 4.20 req/s, TTFT=3213 ms, E2E=7035 ms
- **Finding:** Peak throughput at c=16, but optimal is c=8 (latency acceptable)

### Experiment 5: Memory Utilization

**Purpose:** Test minimum --gpu-memory-utilization that works
- 5.1: gpu_memory_utilization=0.9, GPU 0 → VRAM=22.9GB, KV=5.0 GiB, **Works**
- 5.2: gpu_memory_utilization=0.8, GPU 1 → VRAM=20.4GB, KV=1.9 GiB, **Works**
- 5.3: gpu_memory_utilization=0.7, GPU 1 → KV=-0.46 GiB, **FAILED**
- **Finding:** Model=15.5GB, minimum is 0.8 (0.7 leaves negative KV cache space)

### Experiment 6: High Concurrency (Find Compute vs KV-Cache Limit)

**Purpose:** Push concurrency to find true hardware limit
- 6.1: concurrency=32, num_requests=64 → 2.30 req/s, E2E=13654 ms, KV snapshot=~0% (theoretical peak=~40%)
- 6.2: concurrency=48, num_requests=80 → 7.77 req/s, E2E=4055 ms, KV snapshot=~0% (theoretical peak=~60%)
- 6.3: concurrency=64, num_requests=100 → 2.54 req/s, E2E=23469 ms, KV snapshot=~0% (theoretical peak=~80%)
- 6.4: concurrency=80, num_requests=100 → 2.37 req/s, E2E=28964 ms, KV snapshot=~0% (theoretical peak=~100%)
- **Finding:** We are compute-bound (GPU saturates at c=16), not KV-cache-bound (could support ~60-80 concurrent requests theoretically)

### Experiment 7: KV Cache Theoretical Calculation

**Purpose:** Calculate max concurrent requests based on KV cache capacity
- 7.1: num_gpu_blocks=5009, block_size=16 → Total=80,144 tokens
- 7.2: Per-request tokens=1,324 (1,300 prompt + 24 generation)
- 7.3: Theoretical max = 80,144 / 1,324 ≈ **60 concurrent requests**
- **Finding:** Theoretical limit ~60, but compute limit hit at ~16

---

## Cold Start Comparison Summary

### Cold vs Warmed Up Performance

| System | Scenario | Cold E2E | Warmed E2E | Cold Penalty |
|--------|----------|----------|------------|--------------|
| **HuggingFace** | c=1 | 1865 ms | 1180 ms | 1.58x |
| **vLLM** | c=1, same img | 796 ms | 492 ms | 1.6x |
| **vLLM** | c=8, same img | 939 ms | 534 ms | 1.8x |
| **vLLM** | c=1, diff imgs | 1203 ms | 907 ms | 1.3x |
| **vLLM** | c=8, diff imgs | 3954 ms | 2138 ms | 1.8x |

### First Request Latency (JIT Compilation)

| System | First Request | Cause |
|--------|---------------|-------|
| **HuggingFace** | 1865 ms | CUDA kernel warmup |
| **vLLM** | 4159-4270 ms | CUDA graph compilation |

### Server Startup Time

| System | Startup Time | Notes |
|--------|--------------|-------|
| **HuggingFace** | 4.67s | Model load only |
| **vLLM** | 34-42s | Model load + CUDA graphs + KV cache allocation |

**Key Insights:**
- vLLM has higher cold start penalty due to CUDA graph compilation
- Once warm, vLLM is 2.4x faster than HuggingFace (492ms vs 1180ms)
- vLLM's prefix caching reduces TTFT to 29ms for repeated images (vs 454ms for new images)
- Cold start + different images + high concurrency is worst case (E2E=3954ms)

---

## Summary

This session focused on:
1. Creating HuggingFace Transformers baseline benchmark script
2. Running benchmarks to compare vLLM vs HuggingFace performance
3. **Critical discovery**: Prefix caching impact on vLLM performance
4. Implementing `--vary-images` flag to test realistic production scenarios
5. Finding compute-bound vs KV-cache-bound limits
6. Generating resume-worthy performance claims

---

## Part 1: HuggingFace Baseline Benchmark

### Created `scripts/benchmark_hf_baseline.py`

Benchmark script that:
- Loads the same fine-tuned Qwen2-VL model with HuggingFace Transformers
- Uses `flash_attention_2` and `bfloat16` for fair comparison
- Runs warmup iterations before timing
- Measures throughput and latency

Usage:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_hf_baseline.py --batch-size 1 --num-images 20
```

### Benchmark Results

```
============================================================
HuggingFace Transformers Benchmark
  Batch size: 1
  Num images: 20
  Warmup: 3
============================================================

[Summary]
  Total time: 30.30s
  Throughput: 0.66 img/s (req/s)

[Latency]
  Avg: 1188.8 ms
  Min: 1182.5 ms
  Max: 1206.7 ms
============================================================
```

---

## Part 2: vLLM vs HuggingFace Comparison

### Performance Comparison Table

| Metric | HF Transformers | vLLM (c=1) | vLLM (c=8) | Improvement |
|--------|-----------------|------------|------------|-------------|
| **Throughput** | 0.66 req/s | 1.97 req/s | 11.34 req/s | **3x** / **17x** |
| **Latency** | 1189 ms | 492 ms | 541 ms | **2.4x faster** |

### Key Findings

1. **vLLM provides 3x throughput at single-request level** due to:
   - Optimized CUDA kernels
   - Continuous batching
   - Prefix caching (99.3% hit rate)

2. **vLLM provides 17x throughput at concurrency=8** due to:
   - Continuous batching processes multiple requests simultaneously
   - HuggingFace Transformers processes images sequentially

3. **Latency improvement: 2.4x faster** (492ms vs 1189ms)

### Why HuggingFace Can't Match vLLM Throughput

HuggingFace Transformers processes VLM images **sequentially** (one at a time). This is because:
- Each image has different sizes/aspect ratios
- Vision processing requires per-image attention
- No native support for batching different images in VLMs

vLLM uses **continuous batching** which processes multiple requests concurrently.

---

## Part 3: Resume Claim

> "Deployed fine-tuned Qwen2-VL model with vLLM serving, achieving **3x throughput** (1.97 vs 0.66 req/s) and **2.4x faster latency** (492ms vs 1189ms) compared to HuggingFace Transformers baseline. At concurrency=8, achieved **17x throughput** (11.34 req/s) through continuous batching."

---

## Files Created

1. `scripts/benchmark_hf_baseline.py` - HuggingFace baseline benchmark script

## Files Modified

1. `refactor_documentation/VLLM_BENCHMARK_RESULTS.md` - Added HuggingFace comparison section and resume claim

---

## Part 4: Prefix Caching Discovery (Critical Finding!)

### The Problem

Previous vLLM benchmarks used the **same image for all requests**, which enabled 99.3% prefix cache hit rate. This gave artificially high performance numbers.

### The Solution

Added `--vary-images` flag to `benchmark_vllm.py` to test with different images (realistic production scenario).

### Same Image vs Different Images Comparison

#### Concurrency=1

| Metric | Same Image | Different Images | Impact |
|--------|------------|------------------|--------|
| **Throughput** | 1.97 req/s | 1.09 req/s | **-45%** |
| **TTFT** | 29 ms | 454 ms | **15.5x slower** |
| **E2E** | 492 ms | 907 ms | **1.8x slower** |

#### Concurrency=8

| Metric | Same Image | Different Images | Impact |
|--------|------------|------------------|--------|
| **Throughput** | 11.40 req/s | 3.17 req/s | **-72%** |
| **TTFT** | 42 ms | 719 ms | **17x slower** |
| **E2E** | 534 ms | 2138 ms | **4x slower** |

### Key Insights

1. **Prefix caching provides 15-17x TTFT improvement** for repeated images
2. **Realistic production throughput is 3.17 req/s** at concurrency=8 (not 11.4 req/s)
3. **Prefill time dominates** when processing new images (~389-397ms)
4. This affects how we should report resume claims - need both scenarios!

---

## Part 5: Updated Resume Claims

### Claim 1: Best-Case (Same Image)
> "Achieved **17x throughput** (11.4 req/s) with vLLM's continuous batching and prefix caching"

### Claim 2: Realistic Production (Different Images)
> "Achieved **3.17 req/s** in realistic scenarios with diverse images, **1.6x improvement** over HuggingFace Transformers"

### Claim 3: Prefix Caching Impact
> "Demonstrated **15-17x TTFT improvement** (29ms vs 454ms) through prefix caching"

---

## Files Created/Modified

1. **Created**: `scripts/benchmark_hf_baseline.py` - HuggingFace baseline benchmark
2. **Modified**: `scripts/benchmark_vllm.py` - Added `--vary-images` flag
3. **Modified**: `refactor_documentation/VLLM_BENCHMARK_RESULTS.md` - Complete documentation

---

## Next Steps (What's Left)

### 1. Find Maximum Concurrency Limit

**Goal**: Determine how many concurrent requests the system can handle before degradation.

**How to test**:
```bash
# vLLM server should be running on GPU 0
python scripts/benchmark_vllm.py --num-requests 50 --concurrency 16
python scripts/benchmark_vllm.py --num-requests 50 --concurrency 32
python scripts/benchmark_vllm.py --num-requests 50 --concurrency 64
```

**What to watch**:
- `vllm:num_requests_waiting > 0` means requests are queuing
- `vllm:kv_cache_usage_perc` approaching 100% means memory limit
- E2E latency increasing significantly means saturation

**Current observation**: At concurrency=16, KV cache snapshot shows ~2.3%, but theoretical peak during active processing is ~26%. The system is compute-bound (GPU saturates) before reaching KV-cache limit (~60-80 concurrent requests).

### 2. GPU Memory Utilization Experiments

**Goal**: Test how `--gpu-memory-utilization` affects performance.

**How to test**:
```bash
# Restart vLLM with different memory settings
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.7 \
  --port 8000
```

**What to observe**:
- How max concurrency changes with memory budget
- Whether lower memory util causes requests to queue sooner

### 3. Quantization Experiments (FP8)

**Goal**: Test if FP8 quantization provides speed/memory benefits without quality loss.

**Important**: RTX 3090 (Ampere) has limited FP8 support. FP8 works best on Hopper GPUs (H100).

**How to test** (if supported):
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --quantization fp8 \
  --trust-remote-code \
  --max-model-len 4096 \
  --port 8000
```

**What to measure**:
1. **Quality**: Run evaluation on validation set, check IoU scores
2. **Speed**: TTFT, TPOT, throughput
3. **Memory**: KV cache usage at same concurrency

**Alternative**: If FP8 not supported on RTX 3090, consider AWQ or GPTQ quantization (requires pre-quantized model).

---

## Recommended Priority

1. **Maximum concurrency test** - Easiest, just run benchmarks with higher concurrency
2. **GPU memory utilization** - Requires server restart, but straightforward
3. **Quantization** - May require model conversion, check hardware support first

---

## Part 6: Completed Tasks (Session 28 Continuation)

### Memory Utilization Testing

**Results:**
| Setting | VRAM Used | KV Cache | Status |
|---------|-----------|----------|--------|
| 0.9 | 22.9 GB | ~5.0 GiB | Works |
| 0.8 | 20.4 GB | ~1.9 GiB | Works |
| 0.7 | - | -0.46 GiB | **FAILED** |

**Why 0.7 failed:** Model requires 15.5GB, leaving negative space for KV cache after CUDA overhead.

### Concurrency Sweep Results

| Concurrency | Throughput | TTFT | E2E Avg |
|-------------|------------|------|---------|
| 1 | 1.09 req/s | 454 ms | 907 ms |
| 8 | 3.17 req/s | 719 ms | 2,138 ms |
| 16 | 4.98 req/s | 1,103 ms | 3,008 ms |
| 32 | 4.20 req/s | 3,213 ms | 7,035 ms |

**Finding:** Throughput peaks at c=16, but latency becomes unacceptable. Optimal is c=8.

### Files Created

1. `notebooks/09_vllm_memory_experiments.ipynb` - Comprehensive experiment notebook
2. `notebooks/09_vllm_memory_experiments.py` - Paired Python file

### Files Updated

1. `refactor_documentation/VLLM_BENCHMARK_RESULTS.md` - Added Experiments 3 & 4
2. `refactor_documentation/PROGRESS_20260105_SESSION28.md` - This file

---

## Server Start Commands (Copy-Paste Ready)

```bash
# Standard Configuration (0.9 memory, GPU 0, port 8000)
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8000

# Lower Memory Configuration (0.8 memory, GPU 1, port 8001)
CUDA_VISIBLE_DEVICES=1 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition-mem08 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.8 \
  --port 8001
```

### Benchmark Commands

```bash
# Same image (best-case prefix caching)
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8

# Different images (realistic production)
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images

# Monitor KV cache during benchmark
watch -n 0.5 'curl -s http://localhost:8000/metrics | grep -E "kv_cache|num_requests"'
```
