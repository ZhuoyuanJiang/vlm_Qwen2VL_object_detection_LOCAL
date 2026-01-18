# SESSION29 Plan: vLLM Quantization Experiments

**Created**: 2026-01-17
**Related Documentation**: [vlm_quantization_experiments.md](./vlm_quantization_experiments.md) - Contains detailed Q&A, explanations of GPTQ/AWQ/INT8/FP8, GPU limitations, and rationale from planning session.

---

## Objective

**Learning Goal**: Gather comprehensive statistics on how different quantization methods affect VRAM usage, throughput, latency, and accuracy for the fine-tuned Qwen2-VL model. This is an exploratory study to understand the tradeoffs before making any deployment decisions.

## Current Baseline

- **Model**: Fine-tuned Qwen2-VL at `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged`
- **GPU**: RTX 3090 (24GB VRAM, Ampere architecture)
- **Precision**: BF16 (16-bit brain floating point)
- **Performance**: ~3.17 req/s at concurrency=8

### Historical Reference (HuggingFace Evaluation)

The r4-joint model was previously evaluated using HuggingFace inference (not vLLM):

| Model | Mean IoU | Det Rate | IoU>0.5 | IoU>0.7 |
|-------|----------|----------|---------|---------|
| r4-joint | **0.8636** | 100.00% | 92.00% | 88.00% |

**Note**: This result is from HuggingFace evaluation on 50 samples. It serves as historical context but is not directly comparable to vLLM results due to potential preprocessing differences. We will establish a new vLLM baseline for fair quantization comparison.

## What We Want to Learn

For each quantization configuration, collect these statistics:

| Category | Metrics |
|----------|---------|
| **Memory** | Peak VRAM (GB), model size on disk, KV cache capacity |
| **Throughput** | Requests/second at c=1, c=4, c=8 |
| **Latency** | TTFT, TPOT, E2E, prefill time, decode time |
| **Accuracy** | Mean IoU, detection rate, IoU>0.5 rate, IoU>0.7 rate |

---

## Quantization Method Selection (RTX 3090)

| Method | Recommendation | Rationale |
|--------|----------------|-----------|
| **GPTQ INT4** | Primary | ~4x memory savings, best VLM support, Marlin kernel optimized |
| **INT8 W8A16** | Secondary/Fallback | ~2x memory savings, weight-only (simpler than W8A8) |
| **FP8 W8A8** | Skip | RTX 3090 (Ampere 8.6) lacks native FP8 compute |
| **AWQ** | Fallback | Try only if GPTQ has issues with Qwen2-VL |
| **BitsAndBytes** | Skip | Poor vLLM performance (0.36x baseline) |

**Why these choices?** See detailed explanations in [vlm_quantization_experiments.md](./vlm_quantization_experiments.md)

### Critical: LLM-Only Quantization

**For VLMs like Qwen2-VL, quantize ONLY the language model, NOT the vision encoder.**

Vision encoder module names to **exclude** from quantization (need to verify against actual model):
- `model.visual` (likely Qwen2-VL primary vision module)
- `visual.*` (any visual submodules)

```python
# GPTQModel exclusion example - verify module names first!
ignore_patterns = ["re:.*visual.*"]

# To find actual module names, run:
# for name, module in model.named_modules():
#     if 'visual' in name.lower():
#         print(name)
```

**Why exclude vision encoder?**
- Vision encoder is typically already efficient (smaller than LLM)
- Quantizing it provides minimal memory savings
- But can hurt image understanding accuracy significantly

This keeps the vision encoder in BF16/FP16 for accuracy while quantizing the LLM weights.

---

## Implementation Plan

### Phase 1: Setup and Baseline (Tasks 1-4)

**Task 1: Install quantization libraries**
```bash
# In conda environment: /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving
pip install gptqmodel>=2.2.0
pip install pynvml  # For memory monitoring
pip install torchao  # For INT8 W8A16 (fallback option)
```

**Task 2: Establish deterministic validation slice**
- Fix a deterministic validation slice (e.g., first 100 samples from validation set)
- Use `temperature=0` for all inference
- Cache BF16 baseline outputs to JSON for later comparison:
  ```
  /ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_baseline_outputs.json
  ```
- This allows clean IoU drift comparison between quantized and baseline

**Task 3: Run baseline accuracy evaluation via vLLM**
- Start BF16 model with vLLM server (same serving format as quantized runs)
- Evaluate via vLLM API on the fixed validation slice (same 50 samples as prior recipe comparison, or establish new fixed indices)
- **Important**: Use vLLM for baseline too (not HuggingFace) to avoid preprocessing differences
- Store both raw outputs AND parsed bboxes for comparison

**Task 4: Add detailed memory monitoring to benchmark script**

Modify `scripts/benchmark_vllm.py` to record VRAM at three points:
1. **Post-load**: After model loads, before any inference
2. **Post-warmup**: After warmup requests complete
3. **Peak**: Maximum during benchmark run

Also record KV cache usage from vLLM metrics.

```python
# Key additions to benchmark_vllm.py:
import pynvml

@dataclass
class MemoryProfile:
    post_load_mb: float      # After model loads
    post_warmup_mb: float    # After warmup
    peak_mb: float           # Maximum during benchmark
    kv_cache_usage_pct: float  # From vLLM /metrics

def get_memory_usage() -> float:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2
```

### Phase 2: GPTQ INT4 Quantization (Tasks 5-6)

**Task 5: Create GPTQ quantization script**

Create `scripts/quantize_model_gptq.py`:
- Load fine-tuned BF16 model
- Prepare **multimodal** calibration dataset (128 samples with images + text prompts)
- Use calibration prompts that include image tokens so text blocks see them
- **Exclude vision encoder** from quantization

Key parameters:
```python
# GPTQModel configuration
bits = 4
group_size = 128
ignore_patterns = ["re:.*visual.*"]  # Keep vision encoder in BF16

# Output format: gptq_marlin compatible
# This will create quantization_config in config.json
```

Output path: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4`

**Task 6: Verify quantized model loads in vLLM**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  --served-model-name qwen2vl-nutrition \
  --quantization gptq_marlin \
  --dtype half \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

**Format notes:**
- Use `--quantization gptq_marlin` for faster Marlin kernels on RTX 3090 (Ampere)
- **Fallback**: If gptq_marlin fails (kernel incompatibility), use `--quantization gptq` instead
- `--dtype`: Check the quantized model's config.json first. Use `--dtype half` if quantized with FP16, or `--dtype bfloat16` if quantized with BF16. When in doubt, try without --dtype flag first (vLLM often auto-detects).
- Verify `quantization_config` exists in model's config.json after quantization

### Phase 3: Accuracy Evaluation (Task 7)

**Task 7: Create accuracy evaluation script for vLLM server**

Create `scripts/evaluate_vllm_accuracy.py`:
- Reuse `parse_qwen_bbox_output()` from `src/models/inference.py`
- Reuse IoU calculation logic from `src/training/evaluation.py`
- Use **same deterministic validation slice** as baseline (100 samples)
- Use `temperature=0` for deterministic outputs
- Send requests to vLLM server and collect outputs
- Compare against **cached BF16 outputs** (from Task 2)

Metrics to collect:
- Mean/Median IoU (against ground truth)
- Detection rate
- IoU > 0.5 / IoU > 0.7 rates
- **IoU drift vs BF16 baseline** (how much accuracy changed from quantization)
- **Exact match rate** (% of outputs identical to BF16)

### Phase 4: Benchmarking (Tasks 8-9)

**Task 8: Run comprehensive benchmarks**

For each model variant (BF16, GPTQ INT4):
```bash
# Concurrency sweep
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 1 --vary-images
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 4 --vary-images
python scripts/benchmark_vllm.py --num-requests 20 --concurrency 8 --vary-images
```

Collect: throughput, TTFT, TPOT, E2E, VRAM usage

**Task 9: Compile results**

Create comparison table:
| Model | Quant | VRAM (GB) | Throughput (c=8) | Mean IoU | IoU>0.5 |
|-------|-------|-----------|------------------|----------|---------|
| Baseline | BF16 | ~15.5 | 3.17 req/s | TBD | TBD |
| Quantized | GPTQ INT4 | ~4-5 | TBD | TBD | TBD |

### Phase 5: Documentation (Task 10)

**Task 10: Document findings**

Create `refactor_documentation/PROGRESS_YYYYMMDD_SESSION29.md`:
- Follow existing documentation format
- Include configuration, raw results, analysis
- Summary table with comparison
- Production recommendations

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `scripts/benchmark_vllm.py` | Modify | Add detailed VRAM monitoring (post-load, post-warmup, peak) |
| `scripts/quantize_model_gptq.py` | Create | GPTQ INT4 quantization with LLM-only targeting |
| `scripts/evaluate_vllm_accuracy.py` | Create | Accuracy evaluation via vLLM API with BF16 comparison |
| `refactor_documentation/PROGRESS_*_SESSION29.md` | Create | Results documentation |
| `refactor_documentation/vlm_quantization_experiments.md` | Already created | Background rationale (this planning session) |

---

## Experiment Matrix

| Exp | Model | Quantization | Concurrency | Metrics |
|-----|-------|--------------|-------------|---------|
| E1 | BF16 | None | 1, 4, 8 | Baseline perf + accuracy |
| E2 | GPTQ INT4 | gptq_marlin | 1, 4, 8 | Perf + accuracy |
| E3 | INT8 W8A16 | (weight-only) | 1, 4, 8 | Fallback if GPTQ has issues |

**Note on INT8**: Using W8A16 (weight-only) instead of W8A8 because:
- W8A8 requires specific exporters (torchao/compressed-tensors/modelopt)
- W8A16 is simpler and still provides ~2x memory savings
- **Exporter**: Use **torchao** with `--quantization torchao` flag in vLLM

---

## Verification Plan

1. **Quantization success**: Model loads in vLLM without errors
2. **Config verification**: `quantization_config` exists in config.json
3. **Accuracy check**: Compare IoU against cached BF16 baseline outputs
4. **Memory reduction** (weights on disk vs total vLLM VRAM):
   - BF16 baseline: ~14GB weights, ~15-16GB total vLLM VRAM
   - GPTQ INT4 target: ~3.5GB weights, ~8-10GB total vLLM VRAM (includes KV cache + buffers)
   - INT8 W8A16 target: ~7GB weights, ~10-12GB total vLLM VRAM
5. **Throughput**: At least maintain baseline throughput (ideally improve)

---

## Fallback Options

If GPTQ INT4 shows significant accuracy loss (>10% IoU drop):
1. Try GPTQ with smaller group_size (128 -> 64) for better accuracy
2. Try INT8 W8A16 quantization (less aggressive, ~2x savings instead of ~4x)
3. Try AWQ (with caution - known compatibility issues with Qwen2-VL)
4. Accept accuracy tradeoff if memory savings critical

---

## Directory Structure

```
/ssd1/zhuoyuan/vlm_outputs/
├── qwen2vl-nutrition-detection-r4-joint-merged/           # BF16 baseline
├── qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/ # Quantized
└── quantization_experiments/                               # Results
    ├── baseline_outputs.json
    ├── gptq_results.json
    └── comparison.json
```
