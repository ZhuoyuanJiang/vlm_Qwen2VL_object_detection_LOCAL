# Scripts

Training scripts for Qwen2-VL nutrition table detection.

## Files

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_recipe.py` | Flexible recipe-based training with CLI args | **Recommended** for all training runs |
| `train.py` | Simple training script (~150 lines) with hardcoded config | Learning the codebase, quick experiments |
| `run_recipes.sh` | Bash wrapper with simpler syntax | Convenience for running recipes |

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

## Output Paths

All outputs go to `/ssd1/zhuoyuan/vlm_outputs/`:

```
qwen2vl-nutrition-detection-r1-llm-only/      # r1 output
qwen2vl-nutrition-detection-r2-vision-only/   # r2 output
qwen2vl-nutrition-detection-r3-stage1/        # r3 stage 1 (vision)
qwen2vl-nutrition-detection-r3-stage2/        # r3 stage 2 (LLM)
qwen2vl-nutrition-detection-r4-joint/         # r4 output
```
