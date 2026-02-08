# Scripts

Training, serving, benchmarking, and evaluation scripts for Qwen2-VL nutrition table detection.

## Files

### Training

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_recipe.py` | Flexible recipe-based training with CLI args | **Recommended** for all training runs |
| `train.py` | Simple training script (~150 lines) with hardcoded config | Learning the codebase, quick experiments |
| `run_recipes.sh` | Bash wrapper with simpler syntax | Convenience for running recipes |

### Model Preparation

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `merge_lora.py` | Merge LoRA adapters into base model | Before deployment or quantization |
| `quantize_model_gptq.py` | GPTQ INT4 quantization of merged model | Creating quantized model for faster inference |

### Serving

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `serve_vllm.py` | Start standalone vLLM server | Local development, standalone serving |
| `deploy_triton.sh` | Start Triton via Docker with pre-flight checks | Alternative to Dockerfile for manual deployment |
| `setup_triton.py` | Generate Triton config files (`config.pbtxt` + `model.json`) from CLI args | Initial config generation (configs are now maintained by hand in `triton_model_repository/`) |

### Benchmarking

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `benchmark_vllm.py` | Benchmark standalone vLLM (latency, throughput, concurrency sweeps) | Evaluating vLLM serving performance |
| `benchmark_triton.py` | Benchmark Triton HTTP `/generate` endpoint (async, concurrency support) | Evaluating Triton deployment performance |
| `benchmark_hf_baseline.py` | Benchmark HuggingFace Transformers baseline (static batching) | Comparing against vLLM/Triton |

### Evaluation

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `evaluate_vllm_accuracy.py` | Evaluate vLLM accuracy with IoU metrics on validation set | Accuracy evaluation after deployment |
| `establish_baseline.py` | Establish baseline accuracy metrics for quantization comparison | Before/after quantization comparison |
| `validate_triton_accuracy.py` | Validate Triton deployment produces correct outputs | Sanity check after Triton deployment |

### Quick Tests

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `quick_test_vllm_api.py` | Send 1 request to vLLM, print raw output | Smoke test: is the server responding? |
| `quick_test_vllm_with_visualization.py` | Send 1 request + parse bbox + draw on image | Smoke test with visual verification |

---

### Quick Reference (All Scripts)

| Script | What It Does | Example Command |
|--------|---------------|-----------------|
| `train_recipe.py` | Main production training entrypoint for r1/r2/r3/r4 recipes | `python scripts/train_recipe.py --recipe r4-joint --gpu 0,1` |
| `train.py` | Minimal training script for quick codebase learning | `python scripts/train.py` |
| `run_recipes.sh` | Convenience wrapper for common recipe runs | `./scripts/run_recipes.sh r4 0,1` |
| `merge_lora.py` | Merge LoRA adapter into standalone BF16 model | `python scripts/merge_lora.py --adapter-path <adapter_dir> --output-path <merged_dir>` |
| `quantize_model_gptq.py` | Quantize merged BF16 model to GPTQ INT4 | `python scripts/quantize_model_gptq.py --model-path <merged_dir> --output-path <gptq_dir>` |
| `serve_vllm.py` | Launch local vLLM OpenAI-compatible server | `python scripts/serve_vllm.py --gpu 0 --port 8000` |
| `deploy_triton.sh` | Manual Docker-run helper for Triton deployment (alternative path) | `./scripts/deploy_triton.sh --single gptq --detach` |
| `setup_triton.py` | Bootstrap script to generate Triton repo files (`config.pbtxt` + `model.json`) | `python scripts/setup_triton.py --model <merged_dir> --repo <triton_repo_dir>` |
| `benchmark_hf_baseline.py` | Measure HuggingFace baseline throughput/latency | `python scripts/benchmark_hf_baseline.py --batch-size 1 --num-images 20 --warmup 3` |
| `benchmark_vllm.py` | Benchmark standalone vLLM serving | `python scripts/benchmark_vllm.py --port 8000 --num-requests 20 --concurrency 8 --vary-images` |
| `benchmark_triton.py` | Benchmark Triton HTTP/gRPC endpoints | `python scripts/benchmark_triton.py --model qwen2vl_nutrition_gptq_int4 --endpoint http --num-requests 20 --concurrency 4 --vary-images` |
| `establish_baseline.py` | Build BF16 baseline outputs for quantization drift comparison | `python scripts/establish_baseline.py --num-samples 100 --output-dir <out_dir>` |
| `evaluate_vllm_accuracy.py` | Evaluate vLLM model accuracy and optional baseline drift | `python scripts/evaluate_vllm_accuracy.py --num-samples 100 --model-type gptq-int4 --compare-baseline` |
| `validate_triton_accuracy.py` | Compare Triton outputs against vLLM baseline outputs | `python scripts/validate_triton_accuracy.py --model qwen2vl_nutrition_gptq_int4 --baseline <bf16_baseline.json>` |
| `quick_test_vllm_api.py` | One-shot smoke test for vLLM API response | `python scripts/quick_test_vllm_api.py` |
| `quick_test_vllm_with_visualization.py` | vLLM smoke test with local bbox parsing + visualization output | `python scripts/quick_test_vllm_with_visualization.py` |

---

## train_recipe.py (Recommended)

Full-featured training supporting all 4 recipes with command-line arguments.

```bash
# Basic usage
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1

# All options
python scripts/train_recipe.py \
    --recipe r4-joint \    # r1-llm-only, r2-vision-only, r3-two-stage, r4-joint
    --gpu 0,1 \            # GPU IDs (requires 2 GPUs)
    --epochs 3 \           # Number of epochs
    --no-wandb             # Disable W&B logging

# With logging to file
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1 2>&1 | tee training.log
```

### Available Recipes
- `r1-llm-only` - 4-bit quantization + LoRA on LLM only, vision frozen
- `r2-vision-only` - bf16 full fine-tuning of vision encoder, LLM frozen
- `r3-two-stage` - Vision first (r2), then LLM LoRA (r1) from checkpoint
- `r4-joint` - 4-bit quantization + LoRA on both vision and LLM **(best results)**

---

## train.py (AllTokensCollator)

Minimal ~150-line script using `AllTokensCollator`. No CLI arguments.

```bash
python scripts/train.py
```

**Key difference from train_recipe.py:**
- Uses `AllTokensCollator`:
  - Masks: pad tokens + vision tokens (`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`)
  - Trains on: all text tokens (system + user + assistant)
- `train_recipe.py` uses `AssistantOnlyCollator`:
  - Masks: everything except assistant response
  - Trains on: only object category + bbox coordinates (e.g., `nutrition-table` + `(x1,y1),(x2,y2)`)

> **Note**: The main notebook (`fine_tuning_vlm_for_object_detection_trl.ipynb`) includes both collators and runs experiments to compare their performance. `AssistantOnlyCollator` was found to be better for this task.

**To customize:** Edit the `CONFIG` dict at the top of the file.

---

## run_recipes.sh (Convenience)

Bash wrapper around `train_recipe.py` with shorter syntax.

```bash
./scripts/run_recipes.sh r1 0,1      # Run r1-llm-only on GPUs 0,1
./scripts/run_recipes.sh r2 2,3      # Run r2-vision-only on GPUs 2,3
./scripts/run_recipes.sh r3 0,1      # Run r3-two-stage on GPUs 0,1
./scripts/run_recipes.sh r4 0,1      # Run r4-joint on GPUs 0,1
./scripts/run_recipes.sh all         # Run all recipes sequentially
./scripts/run_recipes.sh parallel    # Run r1 and r4 in parallel (4 GPUs)
```

---

## merge_lora.py

Merge LoRA adapter weights into the base Qwen2-VL model to create a standalone model.

```bash
python scripts/merge_lora.py \
    --adapter-path <adapter_dir> \
    --output-path <merged_dir>
```

---

## quantize_model_gptq.py

Quantize a merged BF16 model to GPTQ INT4. Requires `gptqmodel>=2.2.0` (installed in the serving environment).

```bash
python scripts/quantize_model_gptq.py \
    --model-path <merged_dir> \
    --output-path <gptq_dir> \
    --num-calibration-samples 128
```

---

## serve_vllm.py

Start a standalone vLLM OpenAI-compatible server.

```bash
# Serve BF16 model
python scripts/serve_vllm.py --gpu 0

# Serve GPTQ INT4 model
python scripts/serve_vllm.py --gpu 0 \
    --model <gptq_dir> \
    --quantization gptq_marlin --dtype float16

# With custom settings
python scripts/serve_vllm.py --gpu 0 --port 8000 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching
```

---

## deploy_triton.sh

Helper wrapper for starting Triton via Docker with pre-flight checks.

```bash
./scripts/deploy_triton.sh                    # Start with defaults
./scripts/deploy_triton.sh --single gptq      # Single model mode (GPTQ only)
./scripts/deploy_triton.sh --single bf16      # Single model mode (BF16 only)
./scripts/deploy_triton.sh --detach           # Run in background
```

---

## setup_triton.py

Generate Triton config files (`config.pbtxt` + `model.json`) from CLI args.

```bash
python scripts/setup_triton.py \
    --model <merged_dir> \
    --repo <triton_repo_dir>
```

> **Note**: The generated configs in `triton_model_repository/` are now maintained by hand. This script is useful for initial bootstrapping.

---

## benchmark_vllm.py

Benchmark standalone vLLM serving (latency, throughput, concurrency sweeps).

```bash
python scripts/benchmark_vllm.py \
    --port 8000 --num-requests 20 --concurrency 8 --vary-images
```

---

## benchmark_triton.py

Benchmark Triton HTTP `/generate` endpoint (async, concurrency support).

```bash
python scripts/benchmark_triton.py \
    --model qwen2vl_nutrition_gptq_int4 \
    --endpoint http --num-requests 20 --concurrency 4 --vary-images
```

---

## benchmark_hf_baseline.py

Benchmark HuggingFace Transformers baseline with static batching.

```bash
python scripts/benchmark_hf_baseline.py \
    --batch-size 1 --num-images 20 --warmup 3
```

---

## establish_baseline.py

Build BF16 baseline outputs for quantization drift comparison.

```bash
python scripts/establish_baseline.py \
    --num-samples 100 --output-dir <out_dir>
```

---

## evaluate_vllm_accuracy.py

Evaluate vLLM model accuracy with IoU metrics and optional baseline drift comparison.

```bash
python scripts/evaluate_vllm_accuracy.py \
    --num-samples 100 --model-type gptq-int4 --compare-baseline
```

---

## validate_triton_accuracy.py

Compare Triton outputs against vLLM baseline outputs.

```bash
python scripts/validate_triton_accuracy.py \
    --model qwen2vl_nutrition_gptq_int4 \
    --baseline <bf16_baseline.json>
```

---

## quick_test_vllm_api.py

Minimal one-request smoke test — is the vLLM server responding?

```bash
python scripts/quick_test_vllm_api.py
```

---

## quick_test_vllm_with_visualization.py

Smoke test with bbox parsing and visualization output.

```bash
python scripts/quick_test_vllm_with_visualization.py
```

---

## Output Paths

All outputs go to `/ssd1/zhuoyuan/vlm_outputs/`:

```
qwen2vl-nutrition-detection-r1-llm-only/      # r1 output
qwen2vl-nutrition-detection-r2-vision-only/   # r2 output
qwen2vl-nutrition-detection-r3-stage1/        # r3 stage 1 (vision)
qwen2vl-nutrition-detection-r3-stage2/        # r3 stage 2 (LLM)
qwen2vl-nutrition-detection-r4-joint/         # r4 output
```
