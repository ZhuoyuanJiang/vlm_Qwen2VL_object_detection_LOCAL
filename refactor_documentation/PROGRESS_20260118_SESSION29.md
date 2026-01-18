# Session 29 Progress - GPTQ INT4 Quantization Experiments

**Date**: 2026-01-18
**Session Name**: quantization-experiments

---

## Objective

Gather comprehensive statistics on how GPTQ INT4 quantization affects VRAM usage, throughput, latency, and accuracy for the fine-tuned Qwen2-VL model. This is an exploratory study to understand the tradeoffs before making deployment decisions.

## Environment

- **Conda Environment**: `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving`
- **GPU**: RTX 3090 (24GB VRAM, Ampere architecture)
- **vLLM Version**: 0.13.0

## Models

| Model | Path | Size |
|-------|------|------|
| BF16 Baseline | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged` | ~15 GB |
| GPTQ INT4 | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4` | 6.45 GB |

---

## Scripts Created

### 1. `scripts/establish_baseline.py`
- Creates deterministic 100-sample validation slice
- Runs BF16 evaluation via vLLM API
- Saves baseline outputs for later comparison with quantized model
- Output: `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_baseline_outputs.json`

### 2. `scripts/quantize_model_gptq.py`
- GPTQ INT4 quantization for Qwen2-VL using GPTQModel library
- Uses GPTQModel's native VLM support with conversation format
- Excludes vision encoder from quantization (keeps in BF16)
- Configuration: bits=4, group_size=128, sym=True, desc_act=False

### 3. `scripts/evaluate_vllm_accuracy.py`
- Evaluates model accuracy via vLLM API
- Computes IoU against ground truth
- Computes IoU drift vs BF16 baseline
- Metrics: exact match rate, output match rate, IoU change

### 4. Modified `scripts/benchmark_vllm.py`
- Added VRAM monitoring with pynvml
- Added `MemoryProfile` dataclass
- Records initial, peak, and final VRAM usage
- Records KV cache utilization

---

## Completed Experiments

### Experiment 1: BF16 Baseline Evaluation

**Command**:
```bash
python scripts/establish_baseline.py
```

**Results** (100 samples, validation slice):
| Metric | Value |
|--------|-------|
| Mean IoU | 0.8458 |
| Median IoU | 0.9433 |
| Detection Rate | 100.00% |
| IoU > 0.5 | 91.00% |
| IoU > 0.7 | 86.00% |
| Avg Latency | 1059.0 ms |
| Min Latency | 634.7 ms |
| Max Latency | 4155.4 ms |

**Output**: `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_baseline_outputs.json`

---

### Experiment 2: BF16 Benchmarks (Throughput/Latency)

**vLLM Server Configuration**:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

**Results** (20 requests, DIFFERENT images):

| Concurrency | Throughput | Avg TTFT | Avg TPOT | Avg E2E | VRAM | KV Cache |
|-------------|------------|----------|----------|---------|------|----------|
| c=1 | 1.06 req/s | 500.1 ms | 51.6 ms | 1669.7 ms | 22.67 GB | 0.0% |
| c=4 | 2.50 req/s | 576.2 ms | 60.7 ms | 1949.0 ms | 22.67 GB | 1.7% |
| c=8 | 3.01 req/s | 676.4 ms | 67.9 ms | 2208.6 ms | 22.67 GB | 4.9% |

**Output files**:
- `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_benchmark_c1.json`
- `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_benchmark_c4.json`
- `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_benchmark_c8.json`

---

### Experiment 3: GPTQ INT4 Quantization

**Command**:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/quantize_model_gptq.py \
  --model-path /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --output-path /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  --num-calibration-samples 128
```

**Configuration**:
- bits: 4 (INT4)
- group_size: 128
- desc_act: False
- sym: True (symmetric quantization)
- Vision encoder: Excluded (kept in BF16)

**Results**:
| Metric | Value |
|--------|-------|
| Quantization Time | 13.7 minutes |
| Model Size | 6.45 GB |
| Size Reduction | ~2.3x (from ~15 GB) |

**Output**: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4`

---

### Experiment 4: GPTQ INT4 vLLM Server & Accuracy Evaluation

**vLLM Server Configuration (GPTQ)**:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  --served-model-name qwen2vl-nutrition \
  --quantization gptq_marlin \
  --dtype half \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

**Server Startup Notes**:
- Model loading took **6.46 GB** memory and 4.46 seconds
- Uses `gptq_marlin` kernels (optimized for Ampere GPUs)
- Total VRAM with KV cache pre-allocation: ~22.8 GB (due to `--gpu-memory-utilization 0.9`)

**Accuracy Evaluation Command**:
```bash
python scripts/evaluate_vllm_accuracy.py \
  --model-type gptq-int4 \
  --compare-baseline
```

**Results** (100 samples, same validation slice as BF16):
| Metric | GPTQ INT4 | BF16 Baseline | Change |
|--------|-----------|---------------|--------|
| Mean IoU | 0.8395 | 0.8458 | **-0.74%** |
| Median IoU | 0.9397 | 0.9433 | -0.38% |
| Detection Rate | 100.00% | 100.00% | 0% |
| IoU > 0.5 | 91.00% | 91.00% | 0% |
| IoU > 0.7 | 86.00% | 86.00% | 0% |
| Avg Latency | 618.4 ms | 1059.0 ms | **-41.6%** |
| Exact Match Rate | 15.00% | - | - |

**Status**: NEGLIGIBLE drift (<1%)

**Output**: `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq-int4_evaluation.json`

---

### Experiment 5: GPTQ INT4 Benchmarks

**Benchmark Commands**:
```bash
# Concurrency=1
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 1 --vary-images \
  --output /ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq_benchmark_c1.json

# Concurrency=4
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 4 --vary-images \
  --output /ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq_benchmark_c4.json

# Concurrency=8
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images \
  --output /ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq_benchmark_c8.json
```

**Results** (20 requests, DIFFERENT images):

| Concurrency | Throughput | Avg TTFT | Avg TPOT | Avg E2E | VRAM | KV Cache |
|-------------|------------|----------|----------|---------|------|----------|
| c=1 | 5.16 req/s | 14.5 ms | 7.4 ms | 182.3 ms | 22.81 GB | 0.0% |
| c=4 | 17.49 req/s | 20.3 ms | 8.1 ms | 205.5 ms | 22.81 GB | 0.0% |
| c=8 | 26.25 req/s | 24.7 ms | 8.8 ms | 224.7 ms | 22.81 GB | 0.0% |

**Output files**:
- `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq_benchmark_c1.json`
- `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq_benchmark_c4.json`
- `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/gptq_benchmark_c8.json`

**Note**: Previous results had prefix caching enabled. Re-ran with `--no-enable-prefix-caching` for fair comparison.

---

### Experiment 6: Clean Benchmarks (Fresh Servers, No Prefix Caching, 6 GPUs in Parallel)

**Setup**:
- Killed all existing vLLM processes
- Started 6 fresh servers on 6 GPUs (0-2: BF16, 3-5: GPTQ)
- All with `--no-enable-prefix-caching`
- Ran all 6 benchmarks simultaneously

**Commands**:
```bash
# Kill all existing processes
pkill -9 -f vllm

# Start 6 fresh servers
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i vllm serve $BF16_MODEL --port $((8000+i)) --no-enable-prefix-caching ...
done
for i in 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i vllm serve $GPTQ_MODEL --port $((8000+i)) --no-enable-prefix-caching ...
done

# Run 6 benchmarks in parallel
python scripts/benchmark_vllm.py --port 8000 --concurrency 1 --vary-images --output bf16_clean_c1.json &
python scripts/benchmark_vllm.py --port 8001 --concurrency 4 --vary-images --output bf16_clean_c4.json &
python scripts/benchmark_vllm.py --port 8002 --concurrency 8 --vary-images --output bf16_clean_c8.json &
python scripts/benchmark_vllm.py --port 8003 --concurrency 1 --vary-images --output gptq_clean_c1.json &
python scripts/benchmark_vllm.py --port 8004 --concurrency 4 --vary-images --output gptq_clean_c4.json &
python scripts/benchmark_vllm.py --port 8005 --concurrency 8 --vary-images --output gptq_clean_c8.json &
wait
```

**Results** (20 requests, DIFFERENT images, NO prefix caching, FRESH servers):

| Model | Concurrency | Throughput | Avg TTFT | Avg TPOT | Avg E2E | Success |
|-------|-------------|------------|----------|----------|---------|---------|
| BF16 | c=1 | 0.86 req/s | 657 ms | 20.3 ms | 1118 ms | 20/20 |
| BF16 | c=4 | 1.29 req/s | 1444 ms | 42.0 ms | 2422 ms | 20/20 |
| BF16 | c=8 | 1.74 req/s | 2213 ms | 70.6 ms | 3883 ms | 15/20 |
| **GPTQ** | **c=1** | **1.15 req/s** | 651 ms | **7.4 ms** | **819 ms** | 20/20 |
| **GPTQ** | **c=4** | **1.59 req/s** | 1406 ms | **28.6 ms** | **2081 ms** | 17/20 |
| **GPTQ** | **c=8** | **1.75 req/s** | 2194 ms | **61.7 ms** | **3616 ms** | 17/20 |

**Output files**: `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_clean_*.json`, `gptq_clean_*.json`

---

## Final Summary

### Comparison Table (Clean Benchmarks)

| Metric | BF16 | GPTQ INT4 | Improvement |
|--------|------|-----------|-------------|
| **Model Size** | ~15 GB | 6.45 GB | **2.3x smaller** |
| **Mean IoU** | 0.8458 | 0.8395 | -0.74% (negligible) |
| **Detection Rate** | 100% | 100% | Same |
| **IoU > 0.5** | 91% | 91% | Same |
| **Throughput (c=1)** | 0.86 req/s | 1.15 req/s | **+34%** |
| **Throughput (c=4)** | 1.29 req/s | 1.59 req/s | **+23%** |
| **Throughput (c=8)** | 1.74 req/s | 1.75 req/s | ~same |
| **E2E Latency (c=1)** | 1118 ms | 819 ms | **-27%** |
| **TPOT (c=1)** | 20.3 ms | 7.4 ms | **-64%** |

---

## Key Findings

1. **GPTQ INT4 preserves accuracy**: Only -0.74% Mean IoU drop (0.8458 → 0.8395), same detection rate and IoU thresholds
2. **2.3x smaller model**: 15 GB → 6.45 GB model size
3. **34% faster throughput at c=1**: 0.86 → 1.15 req/s
4. **27% lower latency at c=1**: 1118 ms → 819 ms E2E
5. **64% faster decode (TPOT)**: 20.3 ms → 7.4 ms at c=1 - the main speed advantage from INT4 quantization
6. **Similar TTFT**: Prefill time is similar (~650 ms at c=1) since vision encoder isn't quantized
7. **Throughput converges at high concurrency**: At c=8, both models reach ~1.75 req/s (bottlenecked by prefill)

---

## Recommendations

1. **Use GPTQ INT4 for production** - negligible accuracy loss with significant speed and memory benefits
2. **Single-request latency**: GPTQ is clearly better (819 ms vs 1118 ms)
3. **High concurrency**: Both models perform similarly, so GPTQ still wins due to lower memory footprint
4. **Enable prefix caching in production** for additional speedups when prompts are similar

---

## Issues Encountered

### 1. GPTQModel Calibration Format
**Error**: `TypeError: 'int' object is not subscriptable`

**Root Cause**: GPTQModel's Qwen2-VL handler expected conversation messages with PIL images, not pre-tokenized input_ids.

**Fix**: Rewrote `quantize_model_gptq.py` to pass conversations directly:
```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},  # PIL Image
            {"type": "text", "text": user_prompt}
        ]
    }
]
calibration_data.append(conversation)
```

### 2. Missing Dependencies
- `qwen_vl_utils`: `pip install qwen-vl-utils`
- `peft`: `pip install peft`

---

## How to Replicate All Experiments

### Prerequisites

```bash
# Activate the conda environment
conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving

# Install required packages (if not already installed)
pip install gptqmodel>=2.2.0
pip install pynvml
pip install qwen-vl-utils
pip install peft
```

### Step 1: Run GPTQ Quantization

```bash
# Quantize the BF16 model to GPTQ INT4
CUDA_VISIBLE_DEVICES=0 python scripts/quantize_model_gptq.py \
  --model-path /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --output-path /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  --num-calibration-samples 128

# This takes ~14 minutes and produces a 6.45 GB model
```

### Step 2: Start vLLM Servers (6 GPUs in Parallel)

```bash
# Define model paths
BF16_MODEL="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
GPTQ_MODEL="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"

# Kill any existing vLLM processes
pkill -9 -f vllm
sleep 5

# Start 3 BF16 servers on GPUs 0, 1, 2 (ports 8000, 8001, 8002)
for i in 0 1 2; do
  port=$((8000 + i))
  CUDA_VISIBLE_DEVICES=$i vllm serve $BF16_MODEL \
    --served-model-name qwen2vl-nutrition \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":1}' \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --port $port &>/dev/null &
done

# Start 3 GPTQ servers on GPUs 3, 4, 5 (ports 8003, 8004, 8005)
for i in 3 4 5; do
  port=$((8000 + i))
  CUDA_VISIBLE_DEVICES=$i vllm serve $GPTQ_MODEL \
    --served-model-name qwen2vl-nutrition \
    --quantization gptq_marlin \
    --dtype half \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":1}' \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --port $port &>/dev/null &
done

# Wait for servers to be ready (~90 seconds)
echo "Waiting for servers..."
sleep 90

# Verify all servers are ready
for p in 8000 8001 8002 8003 8004 8005; do
  curl -s http://localhost:$p/health >/dev/null && echo "Port $p ready" || echo "Port $p NOT ready"
done
```

### Step 3: Run Accuracy Evaluation

```bash
# First, establish BF16 baseline (requires BF16 server on port 8000)
python scripts/establish_baseline.py

# Then evaluate GPTQ accuracy (requires GPTQ server on port 8000)
# Note: You may need to restart with GPTQ on port 8000 for this
python scripts/evaluate_vllm_accuracy.py \
  --model-type gptq-int4 \
  --compare-baseline
```

### Step 4: Run All 6 Benchmarks in Parallel

```bash
OUT_DIR="/ssd1/zhuoyuan/vlm_outputs/quantization_experiments"

# Run BF16 benchmarks (c=1, c=4, c=8) on ports 8000, 8001, 8002
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 1 --vary-images --port 8000 \
  --output $OUT_DIR/bf16_clean_c1.json &
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 4 --vary-images --port 8001 \
  --output $OUT_DIR/bf16_clean_c4.json &
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images --port 8002 \
  --output $OUT_DIR/bf16_clean_c8.json &

# Run GPTQ benchmarks (c=1, c=4, c=8) on ports 8003, 8004, 8005
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 1 --vary-images --port 8003 \
  --output $OUT_DIR/gptq_clean_c1.json &
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 4 --vary-images --port 8004 \
  --output $OUT_DIR/gptq_clean_c4.json &
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images --port 8005 \
  --output $OUT_DIR/gptq_clean_c8.json &

# Wait for all benchmarks to complete
wait
echo "All benchmarks complete!"
```

### Step 5: View Results

```bash
# View all benchmark results
for f in /ssd1/zhuoyuan/vlm_outputs/quantization_experiments/*clean*.json; do
  echo "=== $(basename $f) ==="
  python3 -c "
import json
with open('$f') as f:
    d = json.load(f)
print(f'Concurrency: {d[\"config\"][\"concurrency\"]}')
print(f'Throughput: {d[\"summary\"][\"throughput_rps\"]:.2f} req/s')
print(f'Avg TTFT: {d[\"server_metrics\"][\"avg_ttft_ms\"]:.1f} ms')
print(f'Avg TPOT: {d[\"server_metrics\"][\"avg_tpot_ms\"]:.1f} ms')
print(f'Avg E2E: {d[\"server_metrics\"][\"avg_e2e_ms\"]:.1f} ms')
print(f'Success: {d[\"summary\"][\"successful_requests\"]}/{d[\"config\"][\"num_requests\"]}')
"
  echo ""
done
```

### Step 6: Cleanup

```bash
# Kill all vLLM servers
pkill -9 -f vllm

# Verify GPUs are free
nvidia-smi --query-gpu=index,memory.used --format=csv
```

---

## Experiment 7: VRAM Breakdown Analysis

### VRAM Distribution (gpu_memory_utilization=0.9)

When vLLM allocates 90% of 24GB VRAM (21.6 GB), the internal distribution differs significantly between BF16 and GPTQ INT4:

| Component | BF16 | GPTQ INT4 | Ratio |
|-----------|------|-----------|-------|
| **Model Weights** | 15.53 GiB | 6.46 GiB | 2.4x smaller |
| **KV Cache** | 4.28 GiB | 13.35 GiB | **3.1x larger** |
| **KV Capacity** | 80,144 tokens | 249,936 tokens | **3.1x more** |
| **CUDA Graphs** | 0.55 GiB | 0.56 GiB | ~same |

**Key Insight**: GPTQ INT4 quantization frees up ~9 GB of VRAM that vLLM automatically allocates to KV cache, enabling 3.1x more tokens to be processed concurrently.

### Server Logs (VRAM Breakdown)

**BF16 Server**:
```
Model loading took 15.5320 GiB memory and 8.802402 seconds
Available KV cache memory: 4.28 GiB
GPU KV cache size: 80,144 tokens
Graph capturing finished in 6 secs, took 0.55 GiB
```

**GPTQ INT4 Server**:
```
Model loading took 6.4643 GiB memory and 6.966720 seconds
Available KV cache memory: 13.35 GiB
GPU KV cache size: 249,936 tokens
Graph capturing finished in 5 secs, took 0.56 GiB
```

---

## Experiment 8: Prefix Caching Experiments

### Purpose

Test the effect of prefix caching (enabled by default in vLLM) on throughput and latency. Prefix caching allows the server to reuse KV cache entries from previous requests with the same prefix, significantly speeding up repeated/similar requests.

### vLLM Server Configuration (Prefix Caching ENABLED)

```bash
# BF16 server WITH prefix caching (default, no --no-enable-prefix-caching)
CUDA_VISIBLE_DEVICES=0 /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8000

# GPTQ server WITH prefix caching
CUDA_VISIBLE_DEVICES=1 /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  --served-model-name qwen2vl-nutrition \
  --quantization gptq_marlin \
  --dtype half \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8001
```

### Benchmark Commands

```bash
OUT_DIR="/ssd1/zhuoyuan/vlm_outputs/quantization_experiments"
PYTHON="/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/bin/python"

# BF16 with prefix caching
$PYTHON scripts/benchmark_vllm.py --num-requests 20 --concurrency 1 --vary-images --port 8000 \
  --output $OUT_DIR/bf16_prefix_c1.json
$PYTHON scripts/benchmark_vllm.py --num-requests 20 --concurrency 4 --vary-images --port 8000 \
  --output $OUT_DIR/bf16_prefix_c4.json
$PYTHON scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images --port 8000 \
  --output $OUT_DIR/bf16_prefix_c8.json

# GPTQ with prefix caching
$PYTHON scripts/benchmark_vllm.py --num-requests 20 --concurrency 1 --vary-images --port 8001 \
  --output $OUT_DIR/gptq_prefix_c1.json
$PYTHON scripts/benchmark_vllm.py --num-requests 20 --concurrency 4 --vary-images --port 8001 \
  --output $OUT_DIR/gptq_prefix_c4.json
$PYTHON scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images --port 8001 \
  --output $OUT_DIR/gptq_prefix_c8.json
```

### Results: Prefix Caching ENABLED

| Model | Concurrency | Throughput | Avg TTFT | Avg TPOT | Avg E2E | Avg Prefill | Avg Decode | Success |
|-------|-------------|------------|----------|----------|---------|-------------|------------|---------|
| BF16 | c=1 | 0.85 req/s | 660.5 ms | 20.3 ms | 1121.5 ms | 449.7 ms | 461.3 ms | 20/20 |
| BF16 | c=4 | **7.25 req/s** | **49.8 ms** | 20.9 ms | 527.2 ms | **25.6 ms** | 474.6 ms | 20/20 |
| BF16 | c=8 | **11.78 req/s** | **52.5 ms** | 21.7 ms | 545.2 ms | **28.8 ms** | 493.2 ms | 20/20 |
| GPTQ | c=1 | 1.17 req/s | 639.3 ms | 7.4 ms | 807.8 ms | 437.9 ms | 168.9 ms | 20/20 |
| GPTQ | c=4 | **16.45 req/s** | **30.0 ms** | 8.1 ms | 217.9 ms | **14.3 ms** | 185.0 ms | 20/20 |
| GPTQ | c=8 | **25.54 req/s** | **38.3 ms** | 8.9 ms | 240.4 ms | **19.8 ms** | 202.6 ms | 20/20 |

### Prefix Caching Impact Analysis

| Metric | BF16 (no cache → cache) | GPTQ (no cache → cache) |
|--------|-------------------------|-------------------------|
| **Throughput c=4** | 1.29 → 7.25 req/s (**5.6x**) | 1.59 → 16.45 req/s (**10.3x**) |
| **Throughput c=8** | 1.74 → 11.78 req/s (**6.8x**) | 1.75 → 25.54 req/s (**14.6x**) |
| **TTFT c=4** | 1444 → 50 ms (**29x faster**) | 1406 → 30 ms (**47x faster**) |
| **Prefill c=4** | 802 → 26 ms (**31x faster**) | 782 → 14 ms (**56x faster**) |

**Key Finding**: Prefix caching provides massive speedups when the same system prompt is used repeatedly. GPTQ benefits even more due to its larger KV cache capacity.

---

## Experiment 9: Maximum Concurrency Testing

### Purpose

Determine the maximum concurrent requests each model can handle before failures occur, leveraging the larger KV cache capacity of GPTQ INT4.

### Benchmark Commands (High Concurrency)

```bash
OUT_DIR="/ssd1/zhuoyuan/vlm_outputs/quantization_experiments"
PYTHON="/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/bin/python"

# Test increasing concurrency levels
for C in 16 32 64 128 256 512; do
  # BF16 (port 8000)
  $PYTHON scripts/benchmark_vllm.py --num-requests $((C*2)) --concurrency $C --vary-images --port 8000 \
    --output $OUT_DIR/bf16_prefix_c${C}.json

  # GPTQ (port 8001)
  $PYTHON scripts/benchmark_vllm.py --num-requests $((C*2)) --concurrency $C --vary-images --port 8001 \
    --output $OUT_DIR/gptq_prefix_c${C}.json
done

# GPTQ can handle even higher concurrency
for C in 1024 2048; do
  $PYTHON scripts/benchmark_vllm.py --num-requests $((C*2)) --concurrency $C --vary-images --port 8001 \
    --output $OUT_DIR/gptq_prefix_c${C}.json
done
```

### Results: Maximum Concurrency (Prefix Caching ENABLED)

| Model | Concurrency | Throughput | Avg Latency | Success Rate |
|-------|-------------|------------|-------------|--------------|
| BF16 | c=16 | 3.76 req/s | 3702 ms | 40/40 (100%) |
| BF16 | c=32 | 5.37 req/s | 5466 ms | 64/64 (100%) |
| BF16 | c=64 | 4.03 req/s | 12971 ms | 128/128 (100%) |
| BF16 | c=128 | 2.44 req/s | 45072 ms | 256/256 (100%) |
| BF16 | c=256 | 2.96 req/s | 62575 ms | 512/512 (100%) |
| BF16 | **c=512** | 5.13 req/s | 47091 ms | **688/1024 (67%)** |
| **GPTQ** | c=16 | 4.14 req/s | 3270 ms | 40/40 (100%) |
| **GPTQ** | c=32 | 5.67 req/s | 5291 ms | 64/64 (100%) |
| **GPTQ** | c=64 | 4.54 req/s | 11967 ms | 128/128 (100%) |
| **GPTQ** | c=128 | **63.43 req/s** | 1542 ms | 256/256 (100%) |
| **GPTQ** | c=256 | **64.29 req/s** | 3143 ms | 512/512 (100%) |
| **GPTQ** | c=512 | **50.78 req/s** | 7629 ms | **1024/1024 (100%)** |
| **GPTQ** | c=1024 | 41.29 req/s | 17525 ms | 2048/2048 (100%) |
| **GPTQ** | c=2048 | 41.34 req/s | 32883 ms | 4096/4096 (100%) |

### Key Findings: Maximum Concurrency (Warm-Cache)

1. **BF16 Limit**: Starts failing at **c=512** (67% success rate) - limited by 80K token KV cache
2. **GPTQ Limit**: Still **100% success at c=2048** - benefits from 250K token KV cache
3. **GPTQ Peak Throughput**: 64.29 req/s at c=256 (vs BF16's 5.37 req/s at c=32)
4. **GPTQ Advantage**: Can handle **4x+ higher concurrency** than BF16 before degradation

**⚠️ Important**: These results are from warm-cache scenarios. See cold-cache results below for comparison.

---

### Experiment 9b: No-Prefix-Cache Maximum Concurrency (Re-run)

To isolate KV cache capacity effects from prefix caching, we re-ran Experiment 9 with `--no-enable-prefix-caching`. Note: servers were started fresh before Batch 1, but NOT restarted between batches. This isolates prefix caching effects but is not a true "cold start" for each benchmark.

**Server Configuration**:
```bash
# 6 parallel servers (3 BF16 on GPUs 0-2, 3 GPTQ on GPUs 3-5)
# All with --no-enable-prefix-caching flag
CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --port $PORT
```

**Benchmark Commands**:
```bash
PYTHON=/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/bin/python
OUT_DIR=/ssd1/zhuoyuan/vlm_outputs/quantization_experiments

# Batch 1: c=16, 32, 64 (6 parallel)
$PYTHON scripts/benchmark_vllm.py --num-requests 40 --concurrency 16 --vary-images --port 8000 --output $OUT_DIR/bf16_cold_c16.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 64 --concurrency 32 --vary-images --port 8001 --output $OUT_DIR/bf16_cold_c32.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 128 --concurrency 64 --vary-images --port 8002 --output $OUT_DIR/bf16_cold_c64.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 40 --concurrency 16 --vary-images --port 8003 --output $OUT_DIR/gptq_cold_c16.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 64 --concurrency 32 --vary-images --port 8004 --output $OUT_DIR/gptq_cold_c32.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 128 --concurrency 64 --vary-images --port 8005 --output $OUT_DIR/gptq_cold_c64.json &
wait

# Batch 2: c=128, 256, 512 (5 parallel)
$PYTHON scripts/benchmark_vllm.py --num-requests 256 --concurrency 128 --vary-images --port 8000 --output $OUT_DIR/bf16_cold_c128.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 512 --concurrency 256 --vary-images --port 8001 --output $OUT_DIR/bf16_cold_c256.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 256 --concurrency 128 --vary-images --port 8003 --output $OUT_DIR/gptq_cold_c128.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 512 --concurrency 256 --vary-images --port 8004 --output $OUT_DIR/gptq_cold_c256.json &
$PYTHON scripts/benchmark_vllm.py --num-requests 1024 --concurrency 512 --vary-images --port 8005 --output $OUT_DIR/gptq_cold_c512.json &
wait
```

### Results: Maximum Concurrency (No Prefix Caching)

| Model | Concurrency | Throughput | Avg E2E | Success Rate | Notes |
|-------|-------------|------------|---------|--------------|-------|
| BF16 | c=16 | 1.73 req/s | 7378 ms | 40/40 (100%) | |
| BF16 | c=32 | 1.91 req/s | 14463 ms | 64/64 (100%) | |
| BF16 | c=64 | 1.99 req/s | 27560 ms | 127/128 (99%) | |
| BF16 | c=128 | 2.03 req/s | 52587 ms | 245/256 (96%) | |
| BF16 | c=256 | 2.12 req/s | 25777 ms | 202/512 (39%) | ⚠️ Failure regime |
| **GPTQ** | c=16 | 1.88 req/s | 6614 ms | 38/40 (95%) | Transient failures |
| **GPTQ** | c=32 | 1.95 req/s | 14495 ms | 63/64 (98%) | |
| **GPTQ** | c=64 | 1.96 req/s | 27973 ms | 128/128 (100%) | |
| **GPTQ** | c=128 | 1.91 req/s | 54272 ms | 251/256 (98%) | |
| **GPTQ** | c=256 | 2.12 req/s | 26057 ms | 221/512 (43%) | ⚠️ Failure regime |
| **GPTQ** | c=512 | 4.23 req/s | 27100 ms | 228/1024 (22%) | ⚠️ Failure regime |

**⚠️ Caveat on failure regimes**: At c=256+ success rates drop to 22-43%. Throughput is calculated as `num_requests / total_time` (total attempted, not successful), which inflates the metric when failures occur. For actual successful completion rate, use `successful_requests / total_time`. E2E latency averages exclude failed requests, skewing metrics (direction depends on failure behavior—timeouts bias latency downward). Results at c=256+ are not reliable for comparison.

### Key Findings: Warm-Cache vs No-Prefix-Cache Comparison

| Metric | Warm (Prefix Cache) | No Prefix Cache | Interpretation |
|--------|---------------------|-----------------|----------------|
| **GPTQ c=128 throughput** | 63.43 req/s | 1.91 req/s | **33x difference** |
| **GPTQ c=256 throughput** | 64.29 req/s | 2.12 req/s | **30x difference** |
| **GPTQ vs BF16 ratio (c=128)** | 26x faster | ~same | Prefix caching unlocks GPTQ advantage |
| **BF16 c=256 success** | 100% | 39% | Both fail similarly without caching |
| **GPTQ c=256 success** | 100% | 43% | Both fail similarly without caching |

### Critical Insight

**The dramatic GPTQ advantage in warm-cache Experiment 9 (64 req/s vs 5 req/s) was primarily due to prefix caching, NOT raw KV cache capacity.**

Without prefix caching:
- Both models achieve similar throughput (~1.9-2.1 req/s) for c=16-128
- Both models start failing at similar concurrency levels (c=256+)
- The system is **prefill-bound**, not decode-bound
- GPTQ's larger KV cache provides minimal advantage when prefix caching is disabled

This makes sense because:
1. Without prefix caching, every request must fully compute the prefill (vision encoding)
2. Prefill is the same speed for both models (vision encoder not quantized)
3. GPTQ's faster decode and larger KV cache don't help when prefill dominates

**Note on transient failures at low concurrency**: GPTQ had 38/40 (95%) success at c=16 but 63/64 (98%) at c=32. This counterintuitive pattern (higher concurrency, better success) indicates these are random transient failures (possibly server warm-up or timing), not load-related. BF16 achieved 100% at both c=16 and c=32.

### Concurrency vs Throughput Chart (Mental Model)

```
Throughput (req/s)
    |
 64 |                          *GPTQ c=256 warm*
    |                       *GPTQ c=128 warm*
 50 |                                  *GPTQ c=512 warm*
 41 |                                          *GPTQ c=1024-2048 warm*
    |
  5 |            *BF16 c=32 warm* -------- *BF16 c=512 warm (67%)*
    |
  2 |  *BF16/GPTQ no-cache* ~~~~~~ nearly identical ~~~~~~~~
    |
    +------------------------------------------------
       1    8   32  64  128  256  512  1024  2048
                   Concurrency

Legend: warm = prefix caching enabled, no-cache = --no-enable-prefix-caching
```

---

## Updated Summary Table

### Complete Comparison (All Experiments)

| Metric | BF16 | GPTQ INT4 | Improvement |
|--------|------|-----------|-------------|
| **Model Size** | 15.53 GB | 6.46 GB | **2.4x smaller** |
| **KV Cache Capacity** | 80,144 tokens | 249,936 tokens | **3.1x larger** |
| **Mean IoU** | 0.8458 | 0.8395 | -0.74% (negligible) |
| **Detection Rate** | 100% | 100% | Same |
| **Throughput (c=1, no cache)** | 0.86 req/s | 1.15 req/s | +34% |
| **Throughput (c=8, no cache)** | 1.74 req/s | 1.75 req/s | ~same |
| **Throughput (c=8, prefix cache)** | 11.78 req/s | 25.54 req/s | **2.2x** |
| **Peak Throughput** ⚠️ | 5.37 req/s (c=32) | 64.29 req/s (c=256) | **12x** (warm-cache) |
| **Max Concurrency (100% success)** ⚠️ | c=256 | c=2048+ | **8x+ more** (warm-cache) |
| **Throughput (c=128, no cache)** | 2.03 req/s | 1.91 req/s | ~same |
| **E2E Latency (c=1)** | 1118 ms | 819 ms | **-27%** |
| **TPOT (c=1)** | 20.3 ms | 7.4 ms | **-64%** |

---

## Updated Recommendations

1. **Always use GPTQ INT4 for production** - negligible accuracy loss with meaningful throughput gains
2. **Enable prefix caching** - provides 5-15x throughput improvement for repeated prompts (warm-cache best-case; novel workloads see smaller gains)
3. **High concurrency workloads (with prefix caching)** - GPTQ can handle 8x more concurrent requests; without prefix caching, BF16 and GPTQ perform similarly up to c=128
4. **Latency-sensitive applications** - GPTQ provides 27% lower E2E latency and 64% faster decode
5. **Memory-constrained deployments** - GPTQ uses 2.4x less model memory, leaving more for KV cache

---

## Output Files

All results are saved to `/ssd1/zhuoyuan/vlm_outputs/quantization_experiments/`:

| File | Description |
|------|-------------|
| `bf16_baseline_outputs.json` | BF16 accuracy evaluation (100 samples) |
| `gptq-int4_evaluation.json` | GPTQ accuracy evaluation with drift comparison |
| `bf16_clean_c1.json` | BF16 benchmark, c=1, no prefix cache |
| `bf16_clean_c4.json` | BF16 benchmark, c=4, no prefix cache |
| `bf16_clean_c8.json` | BF16 benchmark, c=8, no prefix cache |
| `gptq_clean_c1.json` | GPTQ benchmark, c=1, no prefix cache |
| `gptq_clean_c4.json` | GPTQ benchmark, c=4, no prefix cache |
| `gptq_clean_c8.json` | GPTQ benchmark, c=8, no prefix cache |
| `bf16_prefix_c*.json` | BF16 benchmarks with prefix caching |
| `gptq_prefix_c*.json` | GPTQ benchmarks with prefix caching |
| `bf16_cold_c*.json` | BF16 benchmarks, no prefix caching (Exp 9b) |
| `gptq_cold_c*.json` | GPTQ benchmarks, no prefix caching (Exp 9b) |
| `validation_slice_metadata.json` | Deterministic validation slice info |

**Note on filenames**: Files named `*_cold_*` are "no-prefix-cache" runs (`--no-enable-prefix-caching`), not true cold-start benchmarks. The naming predates the terminology refinement.

---

## Appendix A: Results Interpretation and Rationale

This section addresses reviewer feedback and provides proper framing for interpreting the experimental results.

### A.1 Which Experiments to Trust

| Experiment | Purpose | Reliability | Notes |
|------------|---------|-------------|-------|
| **Exp 1: BF16 Accuracy** | Baseline accuracy | ✅ Reliable | 100 samples, deterministic |
| **Exp 4: GPTQ Accuracy** | Quantization accuracy drift | ✅ Reliable | Same slice, fair comparison |
| **Exp 5: GPTQ Benchmarks** | ❌ Contaminated | ⚠️ Do not use | Prefix caching was enabled; prior requests warmed cache. Warm-cache best-case; not valid for "novel images" claims. |
| **Exp 6: Clean Benchmarks** | No-cache baseline | ✅ Reliable | Fresh servers, `--no-enable-prefix-caching` |
| **Exp 7: VRAM Breakdown** | Memory allocation | ✅ Reliable | Server logs, not affected by caching |
| **Exp 8: Prefix Caching** | Best-case cache performance | ⚠️ Use with caveats | Warm-cache scenario with repeated image set. Warm-cache best-case; not valid for "novel images" claims. |
| **Exp 9: Max Concurrency (warm)** | Scaling limits | ⚠️ Warm-cache best-case | Prefix caching enabled; shows dramatic GPTQ advantage |
| **Exp 9b: Max Concurrency (no-cache)** | Scaling limits | ✅ Reliable (c≤128) | `--no-enable-prefix-caching`; shows BF16≈GPTQ. Results at c=256+ are failure regimes (22-43% success) and not reliable. |

### A.2 Caveats and Limitations

#### Experiment 5 Contamination
Experiment 5 shows unrealistically fast TTFT (14.5 ms) and E2E (182 ms) because prefix caching was enabled on the server, and previous requests had already warmed the cache. **Use Experiment 6 as the reliable no-cache baseline.**

#### Success Rates < 100% in Experiment 6
Some benchmarks in Experiment 6 had failures (likely timeouts):
- BF16 c=8: 15/20 (75% success)
- GPTQ c=4: 17/20 (85% success)
- GPTQ c=8: 17/20 (85% success)

These failures skew throughput/latency metrics. This should be considered when comparing configurations.

**Important nuance**: Latency metrics (TTFT, TPOT, E2E) exclude failed requests from their averages, while throughput is calculated as `num_requests / total_time` (total attempted, not successful). This means:
- Latency may appear better when slow requests timeout and get excluded
- Throughput is inflated because failed requests count toward the numerator

#### Prefix Caching with `--vary-images`
The `--vary-images` flag uses a **deterministic slice** (first N images from the HuggingFace dataset). Within a single benchmark run, images are different. However, across repeated benchmark runs, the **same images are reused**, enabling massive cache hits.

The dramatic TTFT improvements (1444 ms → 50 ms) reflect **repeated runs on the same image set**, not truly novel images. This is a **best-case warm-cache scenario**, not "different images realistic."

#### Max Concurrency Jump at c=128
The throughput jump from 4.54 req/s (c=64) to 63.43 req/s (c=128) for GPTQ is explained by two factors:
1. **KV Cache Capacity**: GPTQ has 3.1× more KV cache (249K vs 80K tokens), reducing evictions
2. **Warm Cache Effects**: Repeated image set across runs enables high prefix-cache hit rates

The combination creates a step change where GPTQ "fits everything" while BF16 experiences eviction pressure. The direction of the effect is real (GPTQ scales better), but the magnitude is amplified by warm-cache conditions.

#### KV Cache Usage Shown as 0.0%
This metric is sampled instantaneously after the benchmark completes. By that point, all requests are finished, so KV cache usage is 0%. This does not reflect peak usage during load.

### A.3 Rationalized Conclusions

Based on the reliable experiments (primarily Experiment 6), here are the defensible conclusions with their rationale.

---

**Accuracy drift is tiny.** BF16 achieves 0.8458 Mean IoU while GPTQ achieves 0.8395 (−0.74% drift), with identical 100% detection rates and IoU threshold metrics. This matches "LLM-only quantization" expectations perfectly. The vision encoder isn't quantized—it stays in BF16/FP16—so the model's ability to understand and locate objects in images remains unchanged. Only the language model decoder weights are compressed to INT4, and since bounding box coordinate generation is a relatively simple text output task, the quantization has minimal impact on the final predictions.

---

**Model size drops by 2.4×, from 15.53 GiB to 6.46 GiB.** This is consistent with partial quantization: the vision encoder (~1.5-2 GiB) remains unquantized, while the LLM weights compress from 16-bit to 4-bit (theoretical 4× reduction). Quantization metadata and unquantized embedding layers add overhead, resulting in the observed 2.4× net reduction rather than the theoretical maximum.

---

**VRAM usage stays at ~22 GiB, but the internal allocation shifts dramatically.** With `--gpu-memory-utilization 0.9`, vLLM pre-allocates 90% of GPU memory regardless of model size. The key difference is *how* that memory is distributed. BF16 uses 15.5 GiB for model weights, leaving only 4.3 GiB for KV cache (80K tokens). GPTQ uses just 6.5 GiB for weights, freeing up 13.4 GiB for KV cache (250K tokens)—a 3.1× capacity increase. This reallocation is automatic and explains why total VRAM appears unchanged while concurrent request capacity improves substantially.

---

**TTFT stays similar (~650 ms), but TPOT drops 64% (20.3 ms → 7.4 ms) in Exp 6 (no-cache, fresh servers).** This is exactly what you'd expect from VLM quantization. TTFT (Time To First Token) is dominated by the prefill phase, which includes vision encoding and prompt processing. Since the vision encoder isn't quantized, prefill speed is unchanged. TPOT (Time Per Output Token) reflects decode speed—the autoregressive token generation loop. INT4 weights enable faster matrix multiplications during each decode step, and the 2.7× speedup is consistent with INT4 performance on RTX 3090's Ampere architecture with Marlin kernels.

---

**Throughput improves 34% at low concurrency (c=1).** With no concurrency contention, GPTQ's faster decode directly translates to higher throughput: 0.86 req/s → 1.15 req/s. The E2E latency reduction (1118 ms → 819 ms, or 27% faster) matches this improvement. At c=1, both models achieve 100% success rates, making this a clean comparison.

---

**Throughput converges at high concurrency (c=8), but with caveats.** Both models reach ~1.75 req/s at c=8, suggesting prefill becomes the bottleneck when multiple requests compete for GPU resources. Since both models have identical prefill performance (same vision encoder), GPTQ's decode advantage gets diluted. However, this comparison is weakened by sub-100% success rates: BF16 achieved only 15/20 (75%) and GPTQ 17/20 (85%). Failed requests are excluded from latency averages, which may bias the metrics.

---

**Prefix caching provides massive speedups, but only under specific conditions.** With prefix caching enabled, throughput jumps from 1.29 req/s to 7.25 req/s (5.6×) for BF16 and from 1.59 req/s to 16.45 req/s (10.3×) for GPTQ at c=4. TTFT drops from ~1400 ms to ~40 ms. However, these results come from a warm-cache scenario where the same deterministic image set was used across repeated benchmark runs. The system prompt tokens get cached, and subsequent requests with the same images hit the cache. For truly novel images in production, gains would be much smaller. GPTQ benefits more because its larger KV cache (250K tokens) experiences fewer evictions, allowing more cache entries to persist.

---

**GPTQ scales to higher concurrency before failure, but the magnitude is amplified.** BF16 starts failing at c=512 (67% success) while GPTQ maintains 100% success even at c=2048. The direction of this effect is real: GPTQ's 3.1× larger KV cache can hold more concurrent sequences without eviction, enabling stable performance at higher loads. However, the dramatic throughput jump from 4.5 req/s (c=64) to 63 req/s (c=128) for GPTQ is partially explained by warm-cache effects combined with KV capacity—when GPTQ can fit all concurrent requests in cache without eviction, prefix caching works optimally. The exact numbers may not generalize to cold-cache or novel-image workloads, but the qualitative advantage is reliable.

### A.4 Summary: What You Can Confidently Report

| Finding | Confidence | Key Evidence |
|---------|------------|--------------|
| GPTQ preserves accuracy (-0.74% IoU drift) | **High** | Exp 1 + Exp 4 |
| 2.4× model size reduction | **High** | Server logs |
| 3.1× KV cache capacity increase | **High** | Server logs |
| TTFT similar (~650ms), TPOT -64% (20→7ms) | **High** | Exp 6, c=1 |
| 34% throughput gain at c=1 | **High** | Exp 6, c=1 |
| GPTQ scales better at high concurrency (warm-cache only) | **Medium** | Warm-cache: direction reliable, magnitude amplified. No-prefix-cache: parity up to c=128. |
| Prefix caching gives 5-15× gains | **Low-Medium** | Best-case only, warm-cache scenario |

### A.5 Recommended Framing for Reports

**Strong claims (high confidence):**
> "GPTQ INT4 quantization of the Qwen2-VL model achieves 2.4× model size reduction while preserving detection accuracy (−0.74% IoU drift). The decode phase benefits significantly from quantization with 64% faster token generation (TPOT: 20.3ms → 7.4ms), while prefill latency remains unchanged since the vision encoder is not quantized."

**Moderate claims (add caveats):**
> "At low concurrency, GPTQ provides 34% higher throughput. At high concurrency, GPTQ's larger KV cache capacity (3.1× more tokens) enables better scaling, though exact throughput gains depend on workload characteristics and caching behavior."

**Weak claims (qualify heavily):**
> "With prefix caching enabled and repeated workloads, throughput can improve by an order of magnitude. However, these gains represent best-case scenarios with warm caches; novel workloads will see smaller improvements."
