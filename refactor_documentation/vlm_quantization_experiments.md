# VLM Quantization Experiments - Background & Rationale

**Created**: 2026-01-17
**Context**: Planning session for quantization experiments on fine-tuned Qwen2-VL model
**Server**: vllab8 (RTX 3090 available, RTX 6000 Ada available)
**vLLM Version**: 0.13.0
**Related Plan**: [SESSION29_Plan_vLLM_quantization_experiments.md](./SESSION29_Plan_vLLM_quantization_experiments.md) - Implementation plan with tasks and verification steps

---

## Table of Contents

1. [Objective](#1-objective) (includes Metrics to Collect / What We Want to Learn)
2. [Current Baseline](#2-current-baseline)
3. [Understanding vLLM Quantization](#3-understanding-vllm-quantization)
4. [Quantization Methods Explained](#4-quantization-methods-explained-detailed)
5. [GPU Limitations & Support](#5-gpu-limitations--support)
6. [Experimental Design Rationale](#6-experimental-design-rationale)
7. [Q&A Documentation](#7-qa-documentation)
8. [Final Experiment Plan](#8-final-experiment-plan)
9. [Prompt for Tutor](#9-prompt-for-tutor)

---

## 1. Objective

**Learning Goal**: Gather comprehensive statistics on how different quantization methods affect VRAM usage, throughput, latency, and accuracy for the fine-tuned Qwen2-VL model.

This is an exploratory study to understand the tradeoffs before making any deployment decisions.

### Metrics to Collect / What We Want to Learn

For each quantization configuration, collect these statistics:

| Category | Metrics |
|----------|---------|
| **Memory** | Peak VRAM (GB), model size on disk, KV cache capacity |
| **Throughput** | Requests/second at c=1, c=4, c=8 |
| **Latency** | TTFT, TPOT, E2E, prefill time, decode time |
| **Accuracy** | Mean IoU, detection rate, IoU>0.5 rate, IoU>0.7 rate |

---

## 2. Current Baseline

- **Model**: Fine-tuned Qwen2-VL at `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged`
- **Task**: Nutrition table bounding box detection
- **GPU**: RTX 3090 (24GB VRAM, Ampere architecture, compute capability 8.6)
- **Precision**: BF16 (16-bit brain floating point)
- **Performance**: ~3.17 req/s at concurrency=8 with vLLM

**Important Note on Performance Numbers**: The baseline performance (3.17 req/s) is specific to the RTX 3090 hardware. Different GPUs will have different baseline numbers due to differences in:
- Memory bandwidth (RTX 3090: 936 GB/s)
- Compute capability (RTX 3090: 35.6 FP16 TFLOPS)
- Architecture optimizations

This is why all quantization experiments should be run on the **same GPU** to ensure fair comparison. Performance differences should reflect quantization impact, not hardware differences.

---

## 3. Understanding vLLM Quantization

### Key Insight: vLLM Serves, Not Quantizes

**vLLM can SERVE quantized models** (native support via `--quantization` flag)
**vLLM does NOT quantize models** (doesn't convert BF16 → INT4)

The workflow is:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Your BF16 Model │  →   │ Quantization    │  →   │ vLLM serves     │
│ (fine-tuned)    │      │ Tool            │      │ quantized model │
│                 │      │ (one-time step) │      │ (production)    │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

### Two Types of Quantization in vLLM

1. **Dynamic Quantization (On-the-fly)**
   - FP8: vLLM can quantize on-the-fly with `--quantization fp8`
   - But: Requires GPU compute capability ≥8.9 (Ada Lovelace, Hopper)

2. **Pre-quantized Models**
   - GPTQ, AWQ, INT8 W8A8: Must be quantized beforehand using external tools
   - vLLM just loads these pre-quantized models

### Why External Tools Are Needed

Quantized models have different weight formats:
- **BF16**: Weights stored as 16-bit floats, standard matrix multiplication
- **INT4 (GPTQ)**: Weights packed (8 weights per 32-bit integer), needs special CUDA kernels

The `--quantization` flag tells vLLM which kernels to use (Marlin, ExLlama, etc.).

### Quantization Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **GPTQModel** | GPTQ quantization | Good VLM support, documented for Qwen2-VL |
| **llm-compressor** | Multiple formats | Built by vLLM team, good integration |
| **AutoGPTQ** | GPTQ quantization | Older, widely used |

---

## 4. Quantization Methods Explained (Detailed)

### 4.1 Understanding the Naming Convention

When you see quantization methods like "INT8 W8A8" or "FP8 W8A16", here's what each part means:

| Symbol | Meaning | Example |
|--------|---------|---------|
| **W** | **W**eight precision | W8 = weights stored in 8 bits |
| **A** | **A**ctivation precision | A16 = activations computed in 16 bits |
| **INT** | Integer data type | INT8 = 8-bit integers (-128 to 127) |
| **FP** | Floating point data type | FP8 = 8-bit floats |
| **BF16** | Brain Float 16 | 16-bit float with 8-bit exponent |

**Examples decoded:**
- `W8A8` = 8-bit weights, 8-bit activations (both compressed)
- `W8A16` = 8-bit weights, 16-bit activations (only weights compressed)
- `W4A16` = 4-bit weights, 16-bit activations (aggressive weight compression)

---

### 4.2 What Are Weights vs Activations?

To understand quantization, you need to know what gets quantized:

```
Neural Network Forward Pass:

Input → [Weights] → Activation → [Weights] → Activation → ... → Output
         Layer 1                   Layer 2

Weights: The learned parameters (stored on disk, loaded into GPU memory)
Activations: The intermediate values computed during inference (temporary)
```

**Weights (W)**:
- The model's learned parameters
- Stored permanently in the model file
- Example: A 7B parameter model has 7 billion weight values
- **Quantizing weights** reduces model file size and GPU memory usage

**Activations (A)**:
- Temporary values computed during inference
- Created when input passes through each layer
- Discarded after use
- **Quantizing activations** speeds up computation but can hurt accuracy

**Industry insight**: Weight-only quantization (W4A16, W8A16) is safer and more common because activations are temporary and compressing them can cause numerical errors to accumulate.

---

### 4.3 GPTQ (Generalized Post-Training Quantization)

**What it is**: An algorithm for compressing model weights to 4-bit integers after training.

**How it works**:
1. Load your trained BF16 model
2. Run a small "calibration dataset" through the model (128-256 samples)
3. For each layer, GPTQ finds the optimal way to round weights to 4-bit
4. It compensates for rounding errors by adjusting other weights
5. Save the compressed model

**The key innovation**: GPTQ uses second-order information (Hessian) to minimize accuracy loss when rounding. Instead of naively rounding 0.7 → 1 or 0, it considers how each weight affects the output.

**Technical details**:
```
Original weight: 0.3421875 (stored in 16 bits = 2 bytes)
GPTQ quantized:  3 (stored in 4 bits = 0.5 bytes) + scale factor

To use: reconstructed_weight = 3 * scale + zero_point ≈ 0.34
```

**Group size**: Weights are quantized in groups (e.g., 128 weights share one scale factor). Smaller groups = better accuracy but larger file size.

**Industry usage**:
- **Very popular** for deploying LLMs on consumer GPUs
- Used by: TheBloke models on HuggingFace, llama.cpp ecosystem
- Companies like Hugging Face, Together AI use GPTQ for inference
- Standard choice when you need 4-bit quantization

**vLLM support**: Full support via `--quantization gptq` or `--quantization gptq_marlin` (faster)

---

### 4.4 AWQ (Activation-aware Weight Quantization)

**What it is**: Similar to GPTQ, but focuses on protecting "important" weights.

**How it works**:
1. Analyze which weights have the biggest impact on activations
2. Protect those important weights (don't quantize them as aggressively)
3. Quantize less important weights more aggressively
4. Net result: Same 4-bit size, but accuracy is better preserved

**The key innovation**: Not all weights are equally important. AWQ identifies the ~1% of weights that matter most (based on activation magnitudes) and treats them specially.

**Technical details**:
```
Traditional: Quantize all weights uniformly
AWQ: Scale important weights UP before quantization, then scale DOWN after
     This preserves their precision during the rounding process
```

**Industry usage**:
- Developed by MIT and NVIDIA researchers
- Popular alternative to GPTQ
- Used by: NVIDIA TensorRT-LLM, some HuggingFace models
- Considered slightly better accuracy than GPTQ in some benchmarks

**vLLM support**: Full support via `--quantization awq`

**Caution for our project**: AWQ has reported compatibility issues with Qwen2-VL models in recent vLLM versions.

---

### 4.5 INT8 Quantization (W8A8 vs W8A16)

**What it is**: Quantize weights (and optionally activations) to 8-bit integers.

**Two variants**:
- **W8A8**: Both weights AND activations are 8-bit
- **W8A16**: Only weights are 8-bit, activations stay 16-bit (weight-only)

**How W8A8 works**:
1. Weights are converted from 16-bit float to 8-bit integer
2. During inference, activations are also computed in 8-bit
3. Uses specialized INT8 matrix multiplication (IMMA on NVIDIA GPUs)

**How W8A16 works** (simpler):
1. Weights are converted from 16-bit float to 8-bit integer
2. During inference, activations stay in 16-bit
3. Weights are dequantized on-the-fly for computation

**Technical details**:
```
BF16 multiplication: 0.5 × 0.3 = 0.15 (floating point math)
INT8 multiplication: 64 × 38 = 2432 (integer math, then rescale)

INT8 range: -128 to 127 (256 possible values)
To represent 0.5: scale it → 0.5 × 127 ≈ 64
```

**W8A8 vs W8A16 tradeoffs**:
| Aspect | W8A8 | W8A16 |
|--------|------|-------|
| Memory savings | ~2x | ~2x |
| Speed improvement | Higher (both quantized) | Moderate |
| Accuracy | Lower (activation errors accumulate) | Higher |
| Complexity | Needs specific exporters | Simpler to implement |
| Tools | torchao/compressed-tensors/modelopt | torchao/bitsandbytes |

**Our choice**: W8A16 (weight-only) because it's simpler and provides similar memory savings with better accuracy preservation.

**Why integers?**:
- Integer math is simpler than floating point math
- GPUs have dedicated INT8 cores (Tensor Cores) that are very fast
- Less memory bandwidth needed (8 bits vs 16 bits)

**Industry usage**:
- **Standard in production inference** at large scale
- Used by: Google (TFLite), NVIDIA (TensorRT), Intel (OpenVINO)
- Default quantization for many mobile/edge deployments
- More conservative than 4-bit, widely trusted
- W8A16 is common for LLMs where accuracy matters

**vLLM support**: Supported via pre-quantized models, torchao, or llm-compressor

---

### 4.6 FP8 (8-bit Floating Point)

**What it is**: A new 8-bit data type that keeps floating point representation.

**How it works**:
```
BF16 (16 bits): 1 sign + 8 exponent + 7 mantissa = can represent ±3.4×10³⁸
FP8 E4M3 (8 bits): 1 sign + 4 exponent + 3 mantissa = can represent ±448
FP8 E5M2 (8 bits): 1 sign + 5 exponent + 2 mantissa = wider range, less precision
```

**Why FP8 over INT8?**:
| Aspect | INT8 | FP8 |
|--------|------|-----|
| **Values** | -128 to 127 (uniform spacing) | Floating point (denser near zero) |
| **Small numbers** | Poor (can't represent 0.001) | Good (can represent small values) |
| **Neural network fit** | Okay (requires careful scaling) | Better (matches weight distribution) |

**Example**:
```
Weights in a layer might be: [-0.02, 0.001, 0.15, -0.08, 0.003, ...]
Most values are small, clustered near zero.

INT8: Wastes values representing large numbers we don't have
FP8: More values allocated to small numbers (exponential spacing)
```

**Industry usage**:
- **Newest standard** pushed by NVIDIA for H100/H200
- Used by: NVIDIA (FP8 in Hopper), AMD (MI300), Microsoft Azure
- Becoming the default for large-scale inference in 2024-2025
- Requires newer GPUs (Ada Lovelace, Hopper)

**vLLM support**:
- Full W8A8 on H100/Ada GPUs
- Partial W8A16 on Ampere GPUs (RTX 3090)

---

### 4.7 Method Comparison Summary

| Method | Bits | Memory Savings | Accuracy | Hardware | Industry Adoption |
|--------|------|----------------|----------|----------|-------------------|
| **BF16** | 16 | Baseline | Best | All GPUs | Training standard |
| **GPTQ INT4** | 4 | ~4x | Good | All GPUs | Very high (consumer) |
| **AWQ INT4** | 4 | ~4x | Good+ | All GPUs | High (enterprise) |
| **INT8 W8A8** | 8 | ~2x | Very good | All GPUs | Very high (production) |
| **FP8 W8A8** | 8 | ~2x | Excellent | H100/Ada | Growing (2024+) |

### 4.8 INT8 vs FP8: Detailed Comparison

| Aspect | INT8 | FP8 |
|--------|------|-----|
| **Data Type** | Integer (1, 2, -5) | Float (1.5, -0.25) |
| **Precision** | Lower (integers only) | Higher (decimals preserved) |
| **Accuracy** | Good, slight degradation | Better, closer to original |
| **Hardware** | Wide GPU support | Newer GPUs only (Ada, Hopper) |
| **Data representation** | Integers: 1, 2, -5, 100 | Floats: 1.5, -0.25, 0.001 |
| **Value range** | -128 to 127 (fixed) | Exponential (more near zero) |
| **Precision for small values** | Poor | Good |
| **Hardware support** | Wide (since ~2018) | Newer GPUs only (2022+) |
| **Computational speed** | Fast (INT8 Tensor Cores) | Fast (FP8 Tensor Cores) |
| **Industry maturity** | Very mature | Emerging standard |

**FP8 is generally "better" because** floating point can represent small differences (0.1, 0.01) that integers cannot. But if your GPU doesn't have native FP8 support, it's slower or doesn't work at all.

**When to use INT8**: Widely compatible, proven, safe choice for most deployments
**When to use FP8**: If you have H100/Ada and want best accuracy at 8-bit

---

## 5. GPU Limitations & Support

### Compute Capability Table

| GPU | Compute Capability | Architecture | FP8 Support |
|-----|-------------------|--------------|-------------|
| RTX 3090 | 8.6 | Ampere | Partial (W8A16 only) |
| RTX 6000 Ada | 8.9 | Ada Lovelace | **Full FP8** |
| H100 | 9.0 | Hopper | **Full FP8** |

### What Works on Each GPU

#### RTX 3090 (Ampere, Compute 8.6)

| Method | Support Level | Recommendation |
|--------|---------------|----------------|
| **GPTQ INT4** | ✅ Full | **Try** |
| **INT8 W8A8** | ✅ Full | **Try** |
| **FP8 W8A16** | ⚠️ Partial (Marlin) | Optional |
| **FP8 W8A8** | ❌ None | **Skip** |

#### RTX 6000 Ada (Ada Lovelace, Compute 8.9)

| Method | Support Level | Recommendation |
|--------|---------------|----------------|
| **GPTQ INT4** | ✅ Full | Try |
| **INT8 W8A8** | ✅ Full | Try |
| **FP8 W8A16** | ✅ Full | Try |
| **FP8 W8A8** | ✅ Full | **Only works here** |

---

## 6. Experimental Design Rationale

### The Problem: GPU Confounding Variable

Different GPUs have different performance characteristics:

| GPU | Memory Bandwidth | FP16 TFLOPS |
|-----|------------------|-------------|
| RTX 3090 | 936 GB/s | 35.6 |
| RTX 6000 Ada | 960 GB/s | 91.1 |

If we run GPTQ INT4 on RTX 3090 and FP8 W8A8 on Ada, we **cannot** fairly compare them because:
- Ada is ~2.5x faster in raw compute
- Any speedup might be from the GPU, not the quantization method

### Solution: Single GPU Approach

**Decision: Use RTX 3090 only for now**

This ensures clean, directly comparable results:
- All methods run on identical hardware
- Performance differences are due to quantization, not GPU

**Trade-off**: We skip FP8 W8A8 (which only works on Ada)

**Future work**: Run all experiments again on Ada for FP8 testing

### Final Method Selection for RTX 3090

| Priority | Method | Why |
|----------|--------|-----|
| 1st | **GPTQ INT4** | Best memory savings (4x), mature VLM support |
| 2nd | **INT8 W8A8** | Middle ground, 2x savings, compare with INT4 |
| Skip | **FP8 W8A8** | Doesn't work on RTX 3090 |

---

## 7. Q&A Documentation

### Q1: Why do we need external tools like GPTQModel? Doesn't vLLM support quantization natively?

**Answer**: vLLM supports **serving** quantized models, not **creating** them.

- For FP8: vLLM can do dynamic quantization on-the-fly (with `--quantization fp8`)
- For GPTQ/AWQ/INT8: You must pre-quantize the model using a tool like GPTQModel

The workflow:
1. Run quantization tool ONCE to convert BF16 → INT4
2. vLLM serves the quantized model with `--quantization gptq`

### Q2: Why do people serve official pre-quantized models? Those aren't fine-tuned.

**Answer**: You're right. Pre-quantized models exist for:
- Quick demos/prototyping
- People using base models without fine-tuning
- Saving download bandwidth

**For fine-tuned models**: You MUST quantize your own trained model. This is exactly what we're planning to do.

### Q3: What's the difference between INT8 W8A8, FP8 W8A8, and FP8 W8A16?

**Answer**:

| Method | Weights | Activations | Hardware Requirement |
|--------|---------|-------------|---------------------|
| **INT8 W8A8** | 8-bit integer | 8-bit integer | Most GPUs |
| **FP8 W8A8** | 8-bit float | 8-bit float | Ada (8.9+), Hopper |
| **FP8 W8A16** | 8-bit float | 16-bit float | Ampere (8.6+) partial |

- `W8` = Weights are 8-bit
- `A8` = Activations are 8-bit
- `A16` = Activations stay 16-bit

### Q4: Is FP8 always better than INT8?

**Answer**: Generally yes for accuracy, but depends on hardware support.

FP8 preserves decimal precision (0.1, 0.01) that integers lose. But if your GPU doesn't have native FP8 support, INT8 is faster and more reliable.

### Q5: Why does vLLM need the `--quantization` flag? Can't it auto-detect?

**Answer**: Quantized weights are stored differently and need special CUDA kernels.

| Aspect | BF16 | INT4 (GPTQ) |
|--------|------|-------------|
| **Loading** | Read floats directly | Unpack 4-bit integers |
| **CUDA kernel** | Standard cuBLAS | Marlin/ExLlama |
| **Storage** | Dense | Packed + metadata |

vLLM often auto-detects from `config.json`, but the flag ensures explicit control.

### Q6: If we use different GPUs, won't results be incomparable?

**Answer**: Yes. Different GPUs have different compute capabilities. To ensure fair comparison:

**Option A** (Chosen): Use single GPU (RTX 3090), skip FP8 W8A8
**Option B**: Use both GPUs, compare relative speedups within each GPU

We chose Option A for clean, publishable results.

---

## 8. Final Experiment Plan

### Environment Setup

**IMPORTANT**: All quantization and vLLM serving experiments MUST use the dedicated serving environment:

```bash
conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving
```

This environment has:
- vLLM 0.13.0
- gptqmodel 5.6.12
- pynvml 13.0.1
- torchao 0.15.0
- Python 3.12

### Phase 1: Setup

1. Activate the serving environment:
   ```bash
   conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving
   ```

2. Verify libraries are installed:
   ```bash
   pip list | grep -iE "(vllm|gptq|pynvml|torchao)"
   ```

3. Add memory monitoring to `scripts/benchmark_vllm.py` (already done)

### Phase 2: Baseline

1. Run BF16 baseline accuracy evaluation
2. Store outputs for comparison

### Phase 3: GPTQ INT4 Quantization

1. Create calibration dataset (128 samples from validation set)
2. Quantize model with GPTQModel
3. Save to `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4`

### Phase 4: INT8 W8A8 Quantization

1. Quantize model with llm-compressor
2. Save to `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-int8`

### Phase 5: Benchmarking

For each model (BF16, GPTQ INT4, INT8 W8A8):
- Run accuracy evaluation
- Run throughput/latency benchmarks at c=1, c=4, c=8
- Record VRAM usage

### Phase 6: Documentation

Create comparison table and analysis in `PROGRESS_*_SESSION29.md`

---

## 9. Prompt for Tutor

The following prompt was prepared to ask for guidance:

---

**Subject: Guidance on Serving Quantized VLM Models with vLLM**

**Context:**
- I have a **fine-tuned Qwen2-VL model** (BF16, ~15GB) for nutrition table detection
- Currently serving with **vLLM 0.13.0** on **RTX 3090** (24GB VRAM, Ampere compute 8.6)
- Baseline performance: ~3.17 req/s at concurrency=8

**Goal:**
I want to serve this model with quantization to:
1. Reduce VRAM usage (currently ~15GB)
2. Potentially improve throughput
3. Learn industry-standard practices for quantized model serving

**My Understanding:**
- vLLM supports **loading pre-quantized models** (GPTQ, AWQ) - requires running a quantization tool first
- vLLM supports **FP8 dynamic quantization** (`--quantization fp8`) - but my RTX 3090 has limited FP8 support
- For 4-bit quantization (GPTQ/AWQ), I need to use a tool like GPTQModel to quantize my fine-tuned model first

**My Questions:**
1. What is the standard industry workflow for serving fine-tuned VLM models with quantization on vLLM?
2. On RTX 3090, which quantization method would you recommend: GPTQ INT4, INT8 W8A8, or trying FP8 anyway?
3. Is there a simpler approach I'm missing? (e.g., does vLLM 0.13 have any native quantization for fine-tuned models?)
4. When quantizing a VLM like Qwen2-VL, should I exclude the vision encoder from quantization (only quantize the language model)?

**Environment:**
- vLLM 0.13.0
- GPU: RTX 3090 (compute 8.6) or RTX 6000 Ada (compute 8.9) available
- Fine-tuned model path: /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged
- Model: Qwen2-VL-7B fine-tuned for bounding box detection

---

## References

- [vLLM Quantization Documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [Qwen vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [GPTQModel GitHub](https://github.com/ModelCloud/GPTQModel)
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [Quantizing Qwen2-VL with GPTQModel (Medium)](https://medium.com/@arunsreekuttan1996/quantizing-qwen2-vl-models-with-gptqmodel-a-complete-guide-for-multi-modal-model-compression-and-f329ea18a17b)
