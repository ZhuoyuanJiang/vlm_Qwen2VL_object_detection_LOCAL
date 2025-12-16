# Session 21: Fix OOM & Add Evaluation Section

**Date:** 2024-12-15
**Focus:** Fix OOM errors in recipe training and add demo model evaluation

---

## Problem Statement

When running `run_two_stage_recipe()` in the notebook, training failed with:
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 774.00 MiB.
GPU 0 has a total capacity of 47.50 GiB of which 593.31 MiB is free.
```

**Root Causes Identified:**
1. `run_recipe()` didn't clean up GPU memory after training, causing OOM when running multiple recipes in sequence
2. Previous models from earlier notebook cells (v1/v2 training, evaluation) were still in GPU memory
3. DataParallel fix from Session 20 needed to be applied to the notebook

---

## Changes Made

### 1. GPU Cleanup in `run_recipe()` Function

**Location:** `fine_tuning_vlm_for_object_detection_trl.py`, lines 3378-3387

**Change:** Added cleanup code at the end of `run_recipe()` to free GPU memory after each recipe finishes:

```python
# Cleanup to prevent OOM when running multiple recipes
out = training_args_recipe.output_dir
print(f"   Cleaning up GPU memory...")
del trainer_recipe
del model_recipe
del processor_recipe
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

return out
```

**Why this works:**
- `trainer_recipe`, `model_recipe`, `processor_recipe` are LOCAL variables created inside `run_recipe()`
- Deleting them at the end of one call doesn't affect subsequent calls (each call creates fresh local variables)
- `gc.collect()` triggers Python garbage collection
- `torch.cuda.empty_cache()` releases unused cached memory back to CUDA
- This cleanup runs after EACH recipe, ensuring GPU memory is freed before the next recipe starts

---

### 2. DataParallel Fix for Two-Stage Training

**Location:** `fine_tuning_vlm_for_object_detection_trl.py`, lines 3325-3332

**Change:** Added fix to prevent DataParallel wrapping when loading from checkpoint:

```python
# Fix for two-stage training: prevent DataParallel when loading from checkpoint
# When using device_map="balanced", the model is already distributed across GPUs.
# HuggingFace Trainer may incorrectly wrap it with DataParallel on top, causing
# dtype mismatch errors (float32 activations vs bf16 lm_head).
# Setting _n_gpu=1 prevents DataParallel while preserving device_map parallelism.
if checkpoint_path and torch.cuda.device_count() > 1:
    training_args_recipe._n_gpu = 1
    print("  [Checkpoint fix] Set _n_gpu=1 to prevent DataParallel (device_map parallelism preserved)")
```

**Why this is needed:**
- Session 20 discovered that Stage 2 of two-stage training failed with dtype mismatch
- Root cause: HuggingFace Trainer wrapped the model with DataParallel on top of device_map parallelism
- Setting `_n_gpu=1` tells Trainer not to use DataParallel
- The model is still distributed via `device_map="balanced"` (model parallelism preserved)
- This fix is conditional - only applies when loading from a checkpoint

---

### 3. Cleanup Cell Before Recipe Execution

**Location:** `fine_tuning_vlm_for_object_detection_trl.py`, lines 3436-3466

**Change:** Added a cleanup cell that runs BEFORE the recipe execution cells to clear any previous models from GPU memory:

```python
# --- IMPORTANT: Clean up GPU memory before running recipes ---
# This cell clears any models/trainers from previous cells to prevent OOM errors.

print("Cleaning up GPU memory before running recipes...")
print(f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Delete common variables from previous training/evaluation cells (only if they exist)
vars_to_delete = [
    'trainer', 'trainer_v2', 'trainer_recipe',
    'model', 'model_finetuned', 'model_finetuned_v2', 'merged_model',
    'processor', 'processor_finetuned',
]

deleted = []
for var_name in vars_to_delete:
    if var_name in globals():
        del globals()[var_name]
        deleted.append(var_name)

if deleted:
    print(f"Deleted variables: {deleted}")
else:
    print("No variables to delete (already clean)")

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("Ready to run recipes!")
```

**Why this is needed:**
- The notebook has multiple cells that load models (v1 training, v2 training, evaluation, etc.)
- If user runs those cells before the recipe cells, GPU memory is already full
- This cleanup cell deletes any leftover models and frees GPU memory
- Uses `if var_name in globals()` to safely check if variable exists before deleting

---

### 4. Evaluation Section for Demo Models

**Location:** `fine_tuning_vlm_for_object_detection_trl.py`, lines 3480-3658

**Change:** Added Section 13 "Evaluate Demo Models" with:

#### 4.1 Configuration Cell
```python
DEMO_MODEL_PATHS = {
    "r1-llm-only-demo": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r1-llm-only-demo",
    "r2-vision-only-demo": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r2-vision-only-demo",
    "r3-stage1-demo": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage1-demo",
    "r3-stage2-demo": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage2-demo",
    "r4-joint-demo": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-demo",
}

# Models using LoRA adapters (need to load base model + adapters)
LORA_DEMO_RECIPES = {"r1-llm-only-demo", "r3-stage2-demo", "r4-joint-demo"}

# Models with full fine-tuning (load directly)
FULL_FINETUNE_DEMOS = {"r2-vision-only-demo", "r3-stage1-demo"}
```

#### 4.2 `load_and_evaluate_demo()` Function
- Loads either LoRA models (base + adapters) or full fine-tuned models
- Runs `evaluate_model()` on the validation set
- Prints per-model metrics (Mean IoU, Detection Rate, IoU thresholds)
- Cleans up GPU memory after evaluation

#### 4.3 Evaluation Loop
- Iterates through all demo model paths
- Skips models that don't exist
- Stores metrics in `demo_metrics` dict

#### 4.4 Summary Table
```
================================================================================
DEMO MODEL COMPARISON SUMMARY
================================================================================

Model                     Mean IoU   Det Rate    IoU>0.5    IoU>0.7
--------------------------------------------------------------------------------
r1-llm-only-demo            0.XXXX     XX.XX%     XX.XX%     XX.XX%
r2-vision-only-demo         0.XXXX     XX.XX%     XX.XX%     XX.XX%
...

Best Demo Model: rX-xxx-demo (Mean IoU: 0.XXXX)

Insights:
  - Two-stage (r3) vs LLM-only (r1): +X.XXXX
  - Joint (r4) vs LLM-only (r1): +X.XXXX
```

---

## File Modified

| File | Changes |
|------|---------|
| `fine_tuning_vlm_for_object_detection_trl.py` | 1. GPU cleanup in `run_recipe()` 2. DataParallel fix 3. Cleanup cell before recipes 4. Evaluation section |

---

## How the Recipe Training Flow Now Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLEANUP CELL (NEW)                                                          │
│   - Deletes trainer, model_finetuned, etc. from previous cells              │
│   - gc.collect() + torch.cuda.empty_cache()                                 │
│   - GPU memory freed before recipes start                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ run_two_stage_recipe()                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 1: run_recipe("r2-vision-only", variant_override="r3-stage1-demo")    │
│   - Loads from HuggingFace (no checkpoint_path)                             │
│   - Trains vision encoder (bf16)                                            │
│   - GPU cleanup after training (NEW)                                        │
│   - Returns stage1_dir                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 2: run_recipe("r1-llm-only", checkpoint_path=stage1_dir, ...)         │
│   - Loads from Stage 1 checkpoint                                           │
│   - _n_gpu=1 applied (prevents DataParallel) (NEW - from Session 20)        │
│   - Trains LLM with LoRA                                                    │
│   - GPU cleanup after training (NEW)                                        │
│   - Returns stage2_dir                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ run_recipe("r2-vision-only")  # Standalone                                  │
│   - GPU cleanup after training (NEW)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ run_recipe("r1-llm-only")  # Standalone                                     │
│   - GPU cleanup after training (NEW)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ run_recipe("r4-joint")                                                      │
│   - GPU cleanup after training (NEW)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ EVALUATION SECTION (NEW)                                                    │
│   - Loads each demo model                                                   │
│   - Runs evaluate_model() on validation set                                 │
│   - Displays comparison table                                               │
│   - Shows best model and insights                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```
---

## Testing

After syncing to notebook:
1. Run the cleanup cell before recipe execution
2. Run `output_dir3 = run_two_stage_recipe()` - should complete both stages without OOM
3. Run remaining recipes (r2, r1, r4) - should complete without OOM
4. Run evaluation section - should show comparison table

---

## Session Summary

**Problems Solved:**
1. OOM when running multiple recipes in sequence (cleanup in `run_recipe()`)
2. OOM when running recipes after previous notebook cells (cleanup cell before recipes)
3. DataParallel dtype mismatch in two-stage training (from Session 20, now in notebook)

**New Features:**
1. Demo model evaluation section with comparison table
2. Automatic insights comparing recipes

**Files Modified:**
- `fine_tuning_vlm_for_object_detection_trl.py` (4 changes)
- `fine_tuning_vlm_for_object_detection_trl.ipynb` (synced via jupytext)

Next step: 
- run the `fine_tuning_vlm_for_object_detection_trl.ipynb` to see if the entire notebook can be run in one-go without interruption.