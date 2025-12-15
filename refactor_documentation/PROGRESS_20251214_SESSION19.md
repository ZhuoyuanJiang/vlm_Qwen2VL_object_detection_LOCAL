# Session 19 Progress - December 14, 2024

## Goal
Create `scripts/train_recipe.py` - A standalone script to run training recipes in parallel on different GPUs.

---

## What Was Done

### 1. Created `scripts/train_recipe.py`

A standalone script for running training recipes with command-line arguments:

**Features:**
- Command-line arguments: `--recipe`, `--gpu`, `--epochs`, `--checkpoint`, `--model-id`, `--no-wandb`
- GPU setup BEFORE torch import (supports `--gpu 0,1`, `CUDA_VISIBLE_DEVICES`, or auto-detection)
- Recipe definitions copied from main notebook (TrainingRecipe dataclass, RECIPES dict)
- Helper functions: `load_model_for_recipe()`, `apply_freeze_policy()`, `build_lora_config_for_recipe()`
- Gradient checkpointing enabled for r2 (vision-only) to reduce memory
- TRL 0.22+ compatible (passes `peft_config` to SFTTrainer)

**Output paths (production, 3 epochs):**
- r1-llm-only → `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r1-llm-only`
- r2-vision-only → `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r2-vision-only`
- r3-two-stage Stage 1 → `-r3-stage1`
- r3-two-stage Stage 2 → `-r3-stage2`
- r4-joint → `-r4-joint`

### 2. Created `scripts/run_recipes.sh`

Shell script for easy recipe execution:
```bash
./scripts/run_recipes.sh r1 0,1     # Run r1 on GPUs 0,1
./scripts/run_recipes.sh all        # Run all sequentially
./scripts/run_recipes.sh parallel   # Run r1 and r4 in parallel (4 GPUs)
```

### 3. Updated Notebook Recipe Framework

Modified `fine_tuning_vlm_for_object_detection_trl.py`:
- `run_recipe()` now uses `-demo` suffix for output paths
- Default `num_epochs=1` for quick demonstration
- Added `variant_override` parameter for custom output names

**Demo output paths (1 epoch):**
- r1-llm-only → `-r1-llm-only-demo`
- r2-vision-only → `-r2-vision-only-demo`
- etc.

### 4. Updated `notebooks/04_evaluation_analysis.py`

- Added `ADAPTER_PATH_V1`, `ADAPTER_PATH_V2` definitions (fixes undefined variable errors)
- Added `RECIPE_MODEL_PATHS` configuration
- Added `LORA_RECIPES` and `FULL_FINETUNE_RECIPES` lists
- Added Section 10: Recipe Model Evaluation (loops through all recipes)
- Updated Section 11: Final summary table shows Base + r1-r4

---

## Manual Fixes by User

The user manually made these fixes to `scripts/train_recipe.py`:

1. **Dataset split fix** (lines 327, 331):
   - Changed `dataset["test"]` → `dataset["val"]`
   - Fixes `KeyError: 'test'` (OpenFoodFacts uses 'val' split)

2. **Vision LoRA targets** (line 146):
   - Added `"attn.proj"` to `VISION_LORA_TARGETS`
   - Allows vision LoRA to adapt the attention output projection

3. **Print trainable parameters guard** (line 447):
   - Guarded `trainer_recipe.model.print_trainable_parameters()`
   - Added fallback for non-PEFT models like r2-vision-only

---

## Potential Inconsistencies (Script vs Notebook)

The following differences may exist between `scripts/train_recipe.py` and the notebook demo:

| Item | Script | Notebook |
|------|--------|----------|
| Dataset split | `dataset["val"]` | May still use `dataset["test"]` |
| VISION_LORA_TARGETS | Includes `"attn.proj"` | May not include it |
| print_trainable_parameters | Guarded | May not be guarded |

**TODO:** Sync these changes to the notebook if needed.

---

## Legacy Models (CLEANUP REMINDER)

These folders are from before the recipe framework and are **NOT used** in the current evaluation notebook:

```bash
# Safe to delete when ready:
rm -rf /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora
rm -rf /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora-assistantonly
rm -rf /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-merged
```

**Keep for now:**
- `-v1-all-tokens` (used for collator comparison in evaluation notebook)
- `-v2-assistant-only` (used for collator comparison in evaluation notebook)

---

## Current Model Status

| Model | Path Suffix | Status |
|-------|-------------|--------|
| r1-llm-only | `-r1-llm-only` | Exists |
| r2-vision-only | `-r2-vision-only` | Exists |
| r3-stage1 | `-r3-stage1` | Exists |
| r3-stage2 | `-r3-stage2` | **Missing** (need full r3-two-stage run) |
| r4-joint | `-r4-joint` | Exists |
| v1-all-tokens | `-v1-all-tokens` | Keep (collator comparison) |
| v2-assistant-only | `-v2-assistant-only` | Keep (collator comparison) |

---

## Training Commands

```bash
# Run individual recipes (each needs 2 GPUs)
python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1
python scripts/train_recipe.py --recipe r2-vision-only --gpu 0,1
python scripts/train_recipe.py --recipe r3-two-stage --gpu 0,1
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1

# Or use shell script
./scripts/run_recipes.sh r1 0,1
./scripts/run_recipes.sh all
```

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/train_recipe.py` | Created (new file) |
| `scripts/run_recipes.sh` | Created (new file) |
| `fine_tuning_vlm_for_object_detection_trl.py` | Updated `run_recipe()` with demo suffix |
| `notebooks/04_evaluation_analysis.py` | Added recipe evaluation section |

---

## Notebook Demo Output Paths

| Recipe | Output Directory |
|--------|------------------|
| `run_recipe("r1-llm-only")` | `-r1-llm-only-demo` |
| `run_recipe("r2-vision-only")` | `-r2-vision-only-demo` |
| `run_two_stage_recipe()` | Stage 1: `-r3-stage1-demo`, Stage 2: `-r3-stage2-demo` |
| `run_recipe("r4-joint")` | `-r4-joint-demo` |

**Key code locations:**
- Line 3317: `variant=f"{recipe.name}-demo"` - adds `-demo` suffix
- Line 3352: `run_two_stage_recipe()` definition
- Line 3365-3368: Calls `run_recipe()` which uses shared paths

## FIXED: run_two_stage_recipe Output Paths

**Problem:** `run_two_stage_recipe()` was overwriting r1 and r2 demo outputs.

**Solution (IMPLEMENTED):**
1. Added `variant_override` parameter to `run_recipe()` (line 3283)
2. Uses `variant_name = variant_override if variant_override else f"{recipe.name}-demo"` (line 3301)
3. Updated `run_two_stage_recipe()` to pass `variant_override="r3-stage1-demo"` and `"r3-stage2-demo"` (lines 3367, 3370)

## Next Steps

1. Complete r3-two-stage training (r3-stage2 is missing)
2. Sync manual fixes to notebook if needed
3. Run evaluation notebook to compare all recipes
4. Clean up legacy model folders when ready
