# Session 18 Progress Report

**Date:** 2025-12-14
**Focus:** Diagnostic Verification & 4 Training Recipes Implementation

---

## What Was Accomplished

### Part 1: Model Architecture Inspection (02_model_understanding.ipynb)

Added a new section "2.1 Model Architecture - Layer Names for LoRA Targeting" with:

**New cells inserted after "Model Loading and Configuration" section:**

1. **Markdown cell** - Introduction explaining model structure and why module names matter for LoRA
2. **Code cell** - Helper functions:
   - `list_linear_modules(model, prefix_filter, max_print=50)` - Lists all nn.Linear modules
   - `find_modules_by_pattern(model, pattern, max_print=50)` - Searches for specific patterns
   - `print_all_parameter_names(model, pattern_filter, max_print=50)` - Prints full parameter paths
3. **Code cell** - Prints full model architecture (`print(model)`)
4. **Code cell** - Prints vision encoder architecture (`print(model.visual)`)
5. **Code cell** - Prints LLM decoder architecture (`print(model.model.language_model)`)
6. **Code cell** - Verifies LoRA target module locations (q_proj, qkv, proj)
7. **Code cell** - Prints ALL parameter names with shapes (for freeze/unfreeze checks)
8. **Markdown cell** - Summary table of parameter path conventions

**Model Structure (CORRECTED):**

```
Qwen2VLForConditionalGeneration (model)
└── (model): Qwen2VLModel
    ├── (visual): Qwen2VisionTransformerPretrainedModel
    └── (language_model): Qwen2VLTextModel
└── (lm_head): Linear
```

**Key Finding: Parameter Paths from `named_parameters()`:**

| Component | Parameter Path Pattern |
|-----------|----------------------|
| Vision Encoder Attention | `model.visual.blocks.{i}.attn.qkv.weight` |
| Vision Encoder Attention Proj | `model.visual.blocks.{i}.attn.proj.weight` |
| Vision Encoder MLP | `model.visual.blocks.{i}.mlp.fc1.weight`, `fc2.weight` |
| LLM Attention | `model.language_model.layers.{i}.self_attn.q_proj.weight` |
| LLM MLP | `model.language_model.layers.{i}.mlp.gate_proj.weight` |
| Output Head | `lm_head.weight` |

**Important:** Parameter names use `model.language_model.layers`, NOT `model.layers`!

---

### Part 2: Quantization & LoRA Diagnostics (main file)

**Location:** Lines ~1467-1601 in `fine_tuning_vlm_for_object_detection_trl.py`

Added two new sections:

1. **"DIAGNOSTIC: BF16 vs 4-bit Trainable Parameters"**
   - `summarize_trainables_by_area()` function - Groups trainable params by visual/llm/lm_head/other
   - Commented-out code to compare bf16 vs 4-bit loading (run once for diagnostics)

2. **"DIAGNOSTIC: LoRA Target Verification"**
   - Commented-out code to temporarily apply `get_peft_model()` for inspection
   - Verifies LoRA targets only LLM layers (not vision encoder)
   - **WARNING:** This is for inspection only, NOT for training!

---

### Part 3: Training Recipes Framework (main file)

**Location:** Lines ~3125-3400 in `fine_tuning_vlm_for_object_detection_trl.py`

Added complete recipe framework:

1. **Target Module Constants:**
   ```python
   LLM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
   VISION_LORA_TARGETS = ["qkv", "fc1", "fc2"]  # Omitting "proj" to avoid conflicts
   ```

2. **TrainingRecipe Dataclass:**
   - `name`, `quantized`, `train_vision`, `train_llm`
   - `lora_target_modules`, `learning_rate`, `description`

3. **RECIPES Dictionary with 4 recipes:**
   - `r1-llm-only` - Baseline: LLM-only QLoRA (4-bit), vision frozen
   - `r2-vision-only` - Vision encoder full fine-tuning, LLM frozen (bf16)
   - `r3-two-stage` - Two-stage: Vision first (r2) → LLM (r1) from checkpoint
   - `r4-joint` - Joint: Vision LoRA + LLM LoRA together

4. **Helper Functions:**
   - `load_model_for_recipe(model_id, recipe, checkpoint_path)` - Loads model with/without quantization
   - `apply_freeze_policy(model, recipe)` - Freezes/unfreezes based on recipe
   - `build_lora_config_for_recipe(recipe)` - Creates LoRA config or None
   - `run_recipe(recipe_key, ...)` - Full pipeline: load → freeze → train → save
   - `run_two_stage_recipe()` - Handles r3 (vision → LLM)

---

## Bug Fixes (Session 18 Continued)

**Critical bugs found and fixed in freeze/unfreeze logic:**

### Bug 1: `summarize_trainables_by_area()` (line ~1503-1516)

**Before (WRONG):**
```python
areas = {"visual": 0, "model.layers": 0, "lm_head": 0, "other": 0}
elif "model.layers" in name or name.startswith("model."):
```

**After (FIXED):**
```python
areas = {"visual": 0, "language_model": 0, "lm_head": 0, "other": 0}
elif "language_model" in name:
```

### Bug 2: `apply_freeze_policy()` (line ~3258-3262)

**Before (WRONG):**
```python
if "model.layers" in name or "lm_head" in name:
```

**After (FIXED):**
```python
if "language_model" in name or "lm_head" in name:
```

**Why these were bugs:** The actual parameter paths are `model.language_model.layers.0...`, NOT `model.layers.0...`. The string `"model.layers"` would never match `"model.language_model.layers"`, so LLM layers would never be unfrozen!

---

## Files Modified

| File | Changes |
|------|---------|
| `notebooks/02_model_understanding.ipynb` | +8 cells for architecture inspection, updated helper functions |
| `fine_tuning_vlm_for_object_detection_trl.py` | +270 lines (diagnostics + recipes), bug fixes in freeze logic |
| `refactor_documentation/SESSION18_Plan.md` | Plan documentation |

---

## What's Left (Session 19)

**Task:** Create `scripts/train_recipe.py` for parallel training

The recipe framework in the main notebook runs sequentially. User wants to run recipes in parallel on different GPUs. Plan created in `SESSION19_Plan.md`:

- Create standalone script with command-line args (`--recipe`, `--gpu`)
- Support both `--gpu` flag and `CUDA_VISIBLE_DEVICES`
- Import from src/ modules (collators, config, loader, etc.)
- Allow running multiple instances in parallel

---

## Important Notes

1. **Vision "proj" conflict:** The vision encoder uses `proj` which might match `o_proj` in LLM. We omit `proj` from `VISION_LORA_TARGETS` to avoid conflicts. User should verify with inspection before using r4-joint.

2. **All recipes use AssistantOnlyCollator** - This is the better collator discovered in earlier sessions.

3. **Diagnostics are commented out** - They're meant to be run once, not every training run.

4. **Do NOT sync .py to .ipynb** - User will do that manually.

5. **Freeze/unfreeze checks must use correct patterns:**
   - Vision: `if ".visual." in name:`
   - LLM: `if "language_model" in name:`
   - NOT `"model.layers"` which doesn't exist in the parameter paths!
