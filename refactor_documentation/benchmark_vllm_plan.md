# Plan: vLLM Performance Benchmarking for Mentor Assignment

## Goal
Learn to collect and interpret vLLM performance metrics (TTFT, TPOT, E2E) and understand key serving knobs.

## Background
- **Model**: Qwen2-VL nutrition detection (fine-tuned)
- **Server**: vllab8 with RTX 3090 (24GB VRAM)
- **Current setup**: vLLM serving on port 8000

---

## Mentor's Assignment Breakdown

### What Mentor Wants You to Learn

| Topic | What to Understand | How to Demonstrate |
|-------|-------------------|-------------------|
| **1. Performance Metrics** | TTFT, TPOT/ITL, E2E latency | Collect and interpret from `/metrics` |
| **2. Prefill vs Decode** | TTFT = prefill + KV cache population; TPOT = decode speed | Explain in writeup |
| **3. Batch Size / Concurrency** | How serving "batch size" = concurrency + token batching | Test at different concurrency levels |
| **4. Metrics Collection** | Use vLLM's `/metrics` Prometheus endpoint | Build script to parse metrics |
| **5. GPU Memory Budgeting** | `--gpu-memory-utilization` controls VRAM allocation | (Later: test different values) |
| **6. Quantization Tradeoffs** | Quality vs Memory vs Speed | (Later: compare BF16 vs FP8) |

### Key Concepts to Explain to Mentor

**LLM Inference has 2 stages:**
1. **Prefill**: Process prompt tokens in parallel, populate KV cache → dominates **TTFT**
2. **Decode**: Generate output tokens one-by-one, reusing KV cache → dominates **TPOT**

**"Batch size" in serving context:**
- Not a fixed N like in training
- = **Concurrency** (# simultaneous requests) + vLLM's internal token batching
- Report results as "at concurrency=4, TTFT was X ms"

### Deliverables for Mentor

1. **Baseline metrics**: TTFT, TPOT, E2E for single request
2. **Concurrency scaling data**: How metrics change from 1→2→4→8 concurrent requests
3. **Writeup**: Explain prefill/decode, what metrics mean, observations

---

## Understanding Experiment Dimensions

### Dimension 1: Concurrency (number of simultaneous requests)
- **Why it matters**: More concurrent requests → higher throughput but potentially higher latency per request
- **Values to test**: 1, 2, 4, 8 concurrent requests
- **What to observe**: How TTFT/TPOT change as load increases

### Dimension 2: Prompt/Input complexity
- For VLMs, this is **image size + text prompt length**
- Our model has fixed input (image + short detection prompt)
- Less variation here for our specific model

### Dimension 3: Output length
- Our model outputs short bboxes (~20-30 tokens)
- Less variation here, but we could test with `max_tokens` settings

### Dimension 4: GPU memory utilization
- `--gpu-memory-utilization` (default 0.9)
- Lower values = less KV cache space = potential bottleneck under load
- **Values to test**: 0.9 (default), 0.7, 0.5

### Dimension 5: Quantization (for later)
- BF16 (current baseline) vs FP8 vs INT8
- Affects accuracy, VRAM, and speed

---

## Experiments to Run (Phase 1: Metrics Focus)

### Experiment 1: Baseline Metrics Collection (Single Request)
**Goal**: Learn to read metrics, establish baseline
- Concurrency: 1
- Run 10-20 requests sequentially
- Collect: TTFT, TPOT, E2E from `/metrics` endpoint
- **Deliverable**: Baseline numbers for single-request latency

### Experiment 2: Concurrency Scaling
**Goal**: Understand how latency changes with load
- Concurrency: 1 → 2 → 4 → 8
- Run 20 requests at each concurrency level
- Collect: TTFT, TPOT, E2E, queue metrics
- **Deliverable**: Table/graph showing latency vs concurrency

---

## Future Experiments (Phase 2: After Phase 1 Complete)

### Experiment 3: GPU Memory Utilization Impact
- Test `--gpu-memory-utilization`: 0.9 vs 0.7 vs 0.5
- Requires restarting vLLM server

### Experiment 4: Quantization Comparison
- Compare BF16 vs FP8
- Measure accuracy + speed + VRAM tradeoffs

---

## Implementation Steps

### Step 1: Verify `/metrics` endpoint works
```bash
curl http://localhost:8000/metrics | grep vllm
```
Understand what metrics are available.

### Step 2: Create benchmark script (`scripts/benchmark_vllm.py`)
- Send requests to vLLM API with timing
- Parse `/metrics` endpoint for TTFT, TPOT, E2E
- Support concurrent requests using asyncio/threading
- Output results to CSV

### Step 3: Run Experiment 1 (Baseline)
- Run 10-20 single requests sequentially
- Collect and record baseline TTFT, TPOT, E2E
- Verify results make sense

### Step 4: Run Experiment 2 (Concurrency Scaling)
- Run at concurrency levels: 1, 2, 4, 8
- 20 requests per level
- Record metrics at each level

### Step 5: Analyze and Document
- Create summary table
- Write up findings for mentor
- Save to `refactor_documentation/VLLM_BENCHMARK_RESULTS.md`

---

## Key Metrics to Collect

From `/metrics` endpoint:
```
vllm:time_to_first_token_seconds      # TTFT
vllm:time_per_output_token_seconds    # TPOT/ITL
vllm:e2e_request_latency_seconds      # E2E
vllm:num_requests_running             # Current load
vllm:num_requests_waiting             # Queue pressure
vllm:gpu_cache_usage_perc             # KV cache usage
```

---

## Files to Create

1. `scripts/benchmark_vllm.py` - Main benchmark script
2. `refactor_documentation/VLLM_BENCHMARK_RESULTS.md` - Results documentation

---

## Success Criteria

1. Can collect TTFT, TPOT, E2E metrics programmatically
2. Have baseline numbers for single-request latency
3. Understand how concurrency affects latency (with data!)
4. Can explain prefill vs decode to mentor
5. Have data to discuss with mentor: "At concurrency=X, TTFT was Y ms"
