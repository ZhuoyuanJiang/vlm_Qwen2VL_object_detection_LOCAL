# vLLM Memory & Concurrency Experiment Plan

**Goal**: Understand vLLM memory consumption, find maximum concurrency, and document performance tradeoffs.

**Server**: vllab8 with 8x RTX 3090 (24GB each)
**Current Setup**: GPU 0, `--gpu-memory-utilization 0.9`

---

## Why VRAM Doesn't Change During Requests

vLLM pre-allocates a fixed amount of GPU memory at startup:

```
Memory Layout:
┌─────────────────────────────────────────────────────────────┐
│  Model Weights (~15GB for Qwen2-VL-7B)                      │
├─────────────────────────────────────────────────────────────┤
│  KV Cache Pool (pre-allocated based on --gpu-memory-util)   │
│  ├── Reused across requests (not allocated/freed)           │
│  └── vllm:kv_cache_usage_perc shows utilization             │
├─────────────────────────────────────────────────────────────┤
│  Reserved free space (1 - gpu_memory_utilization)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiment Matrix

### Experiment 1: Find Max Concurrency (Current Settings)

**Settings**: `--gpu-memory-utilization 0.9` (current)

| Test | Concurrency | Image Mode | Purpose |
|------|-------------|------------|---------|
| 1.1 | 1, 2, 4, 8, 16, 32, 64 | Fresh (no cache) | Find throughput curve |
| 1.2 | 1, 2, 4, 8, 16, 32, 64 | Cached (same images) | Find max with caching |

**Metrics to collect**:
- Throughput (req/s)
- TTFT (ms)
- E2E latency (avg, max)
- `vllm:kv_cache_usage_perc` during test
- `vllm:num_requests_waiting` (queue depth)

**How to ensure "fresh" results**:
- Restart vLLM server to clear prefix cache
- OR use images not seen before in the session

### Experiment 2: Memory Utilization Impact

**Goal**: Test if reducing memory reservation affects performance or max concurrency.

| GPU Memory Util | Expected Memory | KV Cache Blocks | Test |
|-----------------|-----------------|-----------------|------|
| 0.9 (current) | ~22GB | ~5000 | Baseline |
| 0.7 | ~17GB | ~2500? | Medium |
| 0.5 | ~12GB | ~1000? | Low |

**For each setting, test**:
- Max concurrency before requests queue
- Throughput at c=1, 4, 8
- Whether model even loads (may fail at 0.5)

### Experiment 3: KV Cache Monitoring

**Goal**: Understand when KV cache saturates.

During high-concurrency tests, capture:
```bash
# Run in loop during benchmark
watch -n 0.5 'curl -s http://localhost:8000/metrics | grep -E "kv_cache|num_requests"'
```

---

## Detailed Test Scripts

### Script 1: Full Concurrency Sweep (Fresh)

```bash
# Restart vLLM to clear cache
pkill -f "vllm serve"
sleep 5

CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8000 &

sleep 60  # Wait for model to load

# Run fresh tests (each uses new images)
for c in 1 2 4 8 16 32; do
  python scripts/benchmark_vllm.py --num-requests $((c*4)) --concurrency $c --vary-images --output results_fresh_c${c}.json
  sleep 5  # Cool down between tests
done
```

### Script 2: Cached Performance Test

```bash
# After fresh tests, images are now cached
# Run same tests again to see cached performance
for c in 1 2 4 8 16 32; do
  python scripts/benchmark_vllm.py --num-requests $((c*4)) --concurrency $c --vary-images --output results_cached_c${c}.json
done
```

### Script 3: Memory Utilization Sweep

```bash
for mem_util in 0.9 0.7 0.5; do
  echo "=== Testing --gpu-memory-utilization $mem_util ==="

  pkill -f "vllm serve"
  sleep 5

  CUDA_VISIBLE_DEVICES=0 vllm serve \
    /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
    --served-model-name qwen2vl-nutrition \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":1}' \
    --gpu-memory-utilization $mem_util \
    --port 8000 &

  sleep 60

  # Test key concurrency levels
  for c in 1 4 8; do
    python scripts/benchmark_vllm.py --num-requests 20 --concurrency $c --vary-images \
      --output results_mem${mem_util}_c${c}.json
  done
done
```

---

## Expected Results Table (to fill in)

### Concurrency vs Performance (Fresh, 0.9 memory)

| Concurrency | Throughput | TTFT | E2E Avg | E2E Max | KV Cache % |
|-------------|------------|------|---------|---------|------------|
| 1 | 1.09 | 454ms | 907ms | - | ? |
| 2 | ? | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? | ? |
| 8 | 3.17 | 719ms | 2138ms | 4298ms | ? |
| 16 | 4.98 | 1103ms | 3008ms | 5667ms | ? |
| 32 | 4.20 | 3213ms | 7035ms | 14118ms | ? |

### Memory Utilization Impact

| Mem Util | VRAM Used | KV Blocks | Max Conc | Throughput@c=8 |
|----------|-----------|-----------|----------|----------------|
| 0.9 | ~22GB | 5009 | ? | 3.17 req/s |
| 0.7 | ~17GB | ? | ? | ? |
| 0.5 | ~12GB | ? | ? | ? |

---

## Key Questions to Answer

1. **What is the true max concurrency** before latency becomes unacceptable (>5s)?
2. **Can we reduce memory to 0.7 or 0.5** without major performance loss?
3. **What does KV cache usage look like** at different concurrency levels?
4. **Fresh vs Cached**: How much does prefix caching help in mixed workloads?

---

## Recommendation for Production

Based on results, recommend:
- Optimal `--gpu-memory-utilization` setting
- Maximum safe concurrency level
- Expected performance bounds
