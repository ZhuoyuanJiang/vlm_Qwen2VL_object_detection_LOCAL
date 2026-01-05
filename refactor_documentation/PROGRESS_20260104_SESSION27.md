# Session 27 Progress - vLLM Performance Benchmarking

**Date**: 2026-01-04
**Session Name**: session27-vllm-benchmarking

## Summary

This session focused on:
1. Deploying the retrained model (with EOS fix) to vLLM on vllab8
2. Verifying the repetition bug fix works in production
3. Learning vLLM performance metrics for mentor assignment
4. Creating benchmarking tools and running experiments

---

## Part 1: Model Deployment and Verification

### Replaced Model on vllab8

Synced the newly trained model (with EOS token fix from Session 26) from vllab15 to vllab8:

```bash
rsync -avP vllab15:/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged/ \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged/
```

### Started vLLM Server

```bash
conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving

CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --port 8000
```

### Verified EOS Token Fix

Tested without the `stop: ["<|box_end|>"]` workaround:

```
Model Output (raw):
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(273,494),(732,620)<|box_end|>
```

**Result**: Model stops naturally after generating bbox. No repetition bug!

### Removed Stop Workaround from Test Scripts

- `scripts/test_vllm_api.py` - removed `"stop": ["<|box_end|>"]`
- `scripts/test_vllm_with_visualization.py` - removed `"stop": ["<|box_end|>"]`

---

## Part 2: vLLM Performance Benchmarking

### Mentor's Assignment

Learn to collect and interpret vLLM performance metrics:

| Topic | What to Learn |
|-------|---------------|
| Performance Metrics | TTFT, TPOT/ITL, E2E latency |
| Prefill vs Decode | TTFT = prefill; TPOT = decode |
| Batch Size / Concurrency | How serving "batch size" = concurrency |
| Metrics Collection | Use `/metrics` Prometheus endpoint |

### Key Concepts Learned

#### LLM Inference Stages

```
Request arrives
     │
     ▼
┌─────────────┐
│   PREFILL   │  ← Process prompt tokens, populate KV cache
│  (parallel) │    Dominates TTFT (Time To First Token)
└─────────────┘
     │
     ▼
┌─────────────┐
│   DECODE    │  ← Generate output tokens one-by-one
│ (sequential)│    Dominates TPOT (Time Per Output Token)
└─────────────┘
     │
     ▼
Response complete
```

#### Metrics Definitions

| Metric | Full Name | What It Measures |
|--------|-----------|------------------|
| **TTFT** | Time To First Token | Latency until first output token (prefill time) |
| **TPOT** | Time Per Output Token | Average time per generated token (per request) |
| **ITL** | Inter-Token Latency | Time between consecutive tokens (per token) |
| **E2E** | End-to-End Latency | Total request time |

#### KV Cache and Prefix Caching

| Metric | Meaning |
|--------|---------|
| **KV Cache Usage** | % of allocated GPU memory used for attention cache |
| **Prefix Cache Hit Rate** | % of prompt tokens that reused cached KV values |

Our results: 2.3% KV cache usage, 99.3% prefix cache hit rate (excellent!)

---

## Part 3: Files Created

### 1. `scripts/benchmark_vllm.py`

Benchmark script that:
- Sends concurrent requests to vLLM
- Collects metrics from `/metrics` endpoint (TTFT, TPOT, ITL, E2E, Prefill, Decode)
- Supports configurable concurrency levels
- Outputs results to console and JSON

Usage:
```bash
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 4
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --output results.json
```

### 2. `notebooks/08_vllm_performance_analysis.py` (.ipynb synced)

Jupyter notebook for:
- Understanding vLLM metrics
- Running benchmarks at different concurrency levels (1, 2, 4, 8)
- Visualizing results (TTFT, TPOT, ITL, Throughput vs Concurrency)
- Generating summary for mentor

### 3. `refactor_documentation/benchmark_vllm_plan.md`

Detailed plan for the benchmarking experiments.

### 4. `refactor_documentation/VLLM_BENCHMARK_RESULTS.md`

Complete benchmark results including:
- Raw outputs from all runs
- Cold start vs warmed up comparison
- Concurrency scaling results
- Metrics interpretation

---

## Part 4: Benchmark Results

### Cold Start vs Warmed Up (Concurrency=1)

| Metric | Cold Start | Warmed Up | Improvement |
|--------|------------|-----------|-------------|
| **Throughput** | 1.14 req/s | 1.97 req/s | **1.7x** |
| **Avg TTFT** | 332.5 ms | 29.0 ms | **11.5x faster** |
| **Avg E2E** | 796.2 ms | 491.4 ms | **1.6x faster** |
| **Max Latency** | 4159.5 ms | 532.6 ms | **7.8x faster** |

Cold start is slow due to:
- CUDA kernel JIT compilation (~4s one-time cost)
- No prefix cache available yet

### Concurrency Scaling (Warmed Up)

| Concurrency | Throughput | TTFT | E2E | Notes |
|-------------|------------|------|-----|-------|
| 1 | 1.97 req/s | 29 ms | 492 ms | Baseline |
| 8 | 11.34 req/s | 52 ms | 541 ms | **5.7x throughput, +10% latency** |

### Server Stats at Concurrency=16

```
Running: 16 reqs
Waiting: 0 reqs
GPU KV cache usage: 2.3%
Prefix cache hit rate: 99.3%
```

**Interpretation**: Plenty of headroom for higher concurrency!

---

## Part 5: Key Learnings

### 1. Nested Event Loops in Jupyter

`asyncio.run()` fails in Jupyter because Jupyter already runs an event loop. Fixed by adding:

```python
import nest_asyncio
nest_asyncio.apply()
```

### 2. Prometheus Histogram Convention

Any histogram metric (e.g., `vllm:inter_token_latency_seconds`) automatically has:
- `*_sum` - total accumulated value
- `*_count` - number of observations
- `*_bucket` - distribution buckets

### 3. num-requests vs concurrency

| Parameter | Meaning |
|-----------|---------|
| `num-requests` | Total requests to send |
| `concurrency` | Max simultaneous requests |

Example: `--num-requests 20 --concurrency 8` sends 20 requests, 8 at a time.

### 4. What Limits Concurrency

- **KV Cache Memory**: Each concurrent request needs GPU memory for attention cache
- **GPU Memory Budget**: `--gpu-memory-utilization 0.9` allocates 90% of VRAM
- At limit: requests queue (`num_requests_waiting > 0`) or get preempted

---

## Part 6: How to Report to Mentor

> "Using vLLM to serve our fine-tuned Qwen2-VL model on RTX 3090:
>
> At **concurrency=1** (after warmup):
> - **TTFT**: 29ms (fast due to prefix caching)
> - **TPOT**: 20ms (~50 tokens/sec decode speed)
> - **E2E**: 492ms per request
> - **Throughput**: ~2 req/s
>
> At **concurrency=8**:
> - **Throughput**: 11.34 req/s (**5.7x improvement**)
> - **E2E**: 541ms (only +10% latency)
>
> The first request takes ~4s due to CUDA compilation, but subsequent requests stabilize at ~500ms. Prefix caching provides 99.3% hit rate, reducing TTFT from 332ms to 29ms."

---

## Files Modified

1. `scripts/test_vllm_api.py` - Removed stop workaround
2. `scripts/test_vllm_with_visualization.py` - Removed stop workaround
3. `refactor_documentation/PROGRESS_20260103_SESSION26.md` - Added vLLM verification results

## Files Created

1. `scripts/benchmark_vllm.py` - vLLM benchmark script
2. `notebooks/08_vllm_performance_analysis.py` - Analysis notebook
3. `notebooks/08_vllm_performance_analysis.ipynb` - Synced notebook
4. `refactor_documentation/benchmark_vllm_plan.md` - Benchmarking plan
5. `refactor_documentation/VLLM_BENCHMARK_RESULTS.md` - Benchmark results
6. `refactor_documentation/PROGRESS_20260104_SESSION27.md` - This file

---

## Next Steps

1. [ ] Test higher concurrency levels (16, 32) to find the limit
2. [ ] Run experiments with `--gpu-memory-utilization` variations
3. [ ] Test quantization (FP8) for speed/memory tradeoffs
4. [ ] Compare with HuggingFace Transformers baseline for "2x speedup" claim
