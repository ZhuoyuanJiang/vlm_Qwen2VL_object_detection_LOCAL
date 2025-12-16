# Session 23 Progress - Evaluation Bug Fixes (r3-two-stage & r4-joint)

**Date**: 2024-12-16

## Summary

Fixed two critical bugs in `notebooks/04_evaluation_analysis.ipynb` that caused incorrect evaluation results for r3-two-stage (low IoU) and r4-joint (complete failure with 0% detection).

---

## Problem Observed (Before Fix)

When running Section 10 (Recipe Model Evaluation), the results were:

| Model | Mean IoU | Detection Rate | Status |
|-------|----------|----------------|--------|
| r1-llm-only | 0.8349 | 100% | OK |
| r2-vision-only | 0.8330 | 100% | OK |
| r3-two-stage | 0.6221 | 100% | **Low IoU** |
| r4-joint | 0.0000 | 0% | **Complete Failure** |

## Results After Fix

After applying the fixes, all models evaluate correctly:

| Model | Mean IoU | Det Rate | IoU>0.5 | IoU>0.7 |
|-------|----------|----------|---------|---------|
| Base (No Fine-tuning) | 0.0981 | 18.00% | 10.00% | 6.00% |
| r1-llm-only | 0.8349 | 100.00% | 86.00% | 82.00% |
| r2-vision-only | 0.8330 | 100.00% | 88.00% | 82.00% |
| r3-two-stage | 0.8366 | 100.00% | 90.00% | 80.00% |
| r4-joint | **0.8636** | 100.00% | 92.00% | 88.00% |

**Best Recipe: r4-joint** (Mean IoU: 0.8636, +780.0% improvement over base)

---

## Bug 1: r3-two-stage Base Model Mismatch

### Symptoms
- IoU was 0.62 instead of expected ~0.83
- Detection rate was 100% (model "worked" but predictions were inaccurate)

### Root Cause

The r3-two-stage model uses a **two-stage training approach**:

1. **Stage 1**: Fine-tune the vision encoder (full fine-tuning) on the original Qwen base
   - Input: `Qwen/Qwen2-VL-7B-Instruct`
   - Output: Complete model saved to `r3-stage1/` (with fine-tuned vision encoder)

2. **Stage 2**: Fine-tune LLM with LoRA on top of the Stage 1 model
   - Input: `r3-stage1/` (NOT the original Qwen base!)
   - Output: Only LoRA adapters saved to `r3-stage2/`

The original evaluation code loaded ALL LoRA models using the same hardcoded base model:
```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,  # Always "Qwen/Qwen2-VL-7B-Instruct" - WRONG for r3!
    ...
)
model = PeftModel.from_pretrained(model, model_path, ...)
```

But `r3-stage2`'s LoRA adapters were trained on `r3-stage1` (which has a fine-tuned vision encoder), not the original Qwen base.

### Helpful Trace-Through

Let me trace through what actually happened:

**What the TRAINING did:**

**Stage 1 Training:**
- Started with: `Qwen/Qwen2-VL-7B-Instruct` (original)
- Trained: Vision encoder (full fine-tuning)
- Saved: Complete model to `r3-stage1/` (this is now a model with fine-tuned vision encoder)

**Stage 2 Training:**
- Started with: `r3-stage1` (the Stage 1 output - has fine-tuned vision)
- Trained: LLM LoRA adapters
- Saved: Only the LoRA adapters to `r3-stage2/`

The proof is in `r3-stage2/adapter_config.json` line 4:
```json
"base_model_name_or_path": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage1"
```

**What the EVALUATION was doing (incorrectly):**

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL_ID, ...)  # Loads original Qwen
model = PeftModel.from_pretrained(model, r3-stage2, ...)  # Applies LLM LoRA
```

This loads the **original Qwen** (not stage1) as the base, then applies the stage2 LoRA.

### Why Performance Degraded

The Stage 2 LLM LoRA was trained to interpret vision features from the **fine-tuned** vision encoder. When evaluation loaded the **original** (untrained) vision encoder instead:

- The LLM LoRA received mismatched vision features
- The model could still detect objects (100% detection rate) because the LLM LoRA was applied correctly
- But predictions were less accurate because vision features were wrong → IoU dropped from ~0.83 to 0.62

### Fix

Read `adapter_config.json` to dynamically detect and load the correct base model:
```python
adapter_config_path = os.path.join(model_path, "adapter_config.json")
if os.path.exists(adapter_config_path):
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    actual_base = adapter_config.get("base_model_name_or_path", BASE_MODEL_ID)
```

---

## Bug 2: r4-joint GPU Memory Exhaustion

### Symptoms
- All 50 samples failed with error: `"q must be on CUDA"`
- Warning: `"Some parameters are on the meta device because they were offloaded to the cpu"`
- Warning: `"Current model requires 7504 bytes of buffer for offloaded layers"`
- Result: 0% detection rate, 0.00 mean IoU

### Root Cause

**Memory Accumulation**: After evaluating multiple models (v1, v2, base, r1, r2, r3), GPU memory became fragmented. Even with cleanup after each evaluation:
```python
del model
del processor
torch.cuda.empty_cache()
```

Memory didn't fully release because:
1. **No `gc.collect()`**: Python's garbage collector hadn't run, so objects referencing CUDA tensors still existed
2. **No `torch.cuda.synchronize()`**: Async CUDA operations may not have completed before cache was cleared
3. **Cleanup only AFTER evaluation**: No cleanup BEFORE loading the next model

**r4-joint Has More Parameters**: r4-joint has 11 LoRA target modules (including vision encoder modules: `qkv`, `fc1`, `fc2`, `attn.proj`) vs r1's 7 modules (LLM only). This requires more GPU memory.

**FlashAttention Incompatibility**: When GPU memory was insufficient, `device_map="auto"` offloaded some layers to CPU. But the model uses `attn_implementation="flash_attention_2"` which requires ALL tensors on CUDA. During inference, attention computation failed because the query tensor was on CPU.

### Fix

Add aggressive GPU memory clearing at the **START** of `load_and_evaluate_recipe()`:
```python
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

This ensures each model evaluation starts with a clean GPU memory state.

---

## Understanding How LoRA Adapters Are Loaded

### Evaluation Uses Adapter-Based Loading (NOT Merged Models)

In the evaluation code, for LoRA models (r1, r3, r4), we load:

```python
# Step 1: Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(actual_base, ...)

# Step 2: Apply adapter on top
model = PeftModel.from_pretrained(model, model_path, ...)
```

This is **adapter-based loading** - the base model and LoRA weights are kept separate. The model works for inference, but it's not "merged" for deployment.

### `PeftModel.from_pretrained()` Explained

**Syntax:**
```python
model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.bfloat16)
```

| Parameter | What it is |
|-----------|------------|
| `model` | The already-loaded base model (e.g., `Qwen2VLForConditionalGeneration`) |
| `model_path` | Path to the directory containing `adapter_model.safetensors` and `adapter_config.json` |
| `torch_dtype` | (Optional) Data type for the adapter weights |

**What this line does:**

1. **Reads `adapter_config.json`** from `model_path` to understand:
   - Which layers have LoRA adapters (`target_modules`)
   - LoRA hyperparameters (`r`, `lora_alpha`, `lora_dropout`)

2. **Loads `adapter_model.safetensors`** which contains the trained LoRA weights (`lora_A` and `lora_B` matrices)

3. **Injects LoRA layers** into the base model at the specified target modules (e.g., `q_proj`, `v_proj`, etc.)

4. **Returns a `PeftModel`** wrapper that:
   - Forwards inputs through: `original_layer + (lora_B @ lora_A) * scaling`
   - Keeps base weights frozen, only LoRA weights are "active"

### Visual Example

```
Before PeftModel.from_pretrained():
┌─────────────────────────┐
│  Base Model (Qwen2VL)   │
│  ├── q_proj (Linear)    │  ← Original weights only
│  ├── v_proj (Linear)    │
│  └── ...                │
└─────────────────────────┘

After PeftModel.from_pretrained():
┌─────────────────────────────────────┐
│  PeftModel (Base + LoRA)            │
│  ├── q_proj                         │
│  │   ├── base_layer (frozen)        │  ← Original weights
│  │   ├── lora_A (trainable)         │  ← Loaded from adapter
│  │   └── lora_B (trainable)         │  ← Loaded from adapter
│  ├── v_proj                         │
│  │   ├── base_layer (frozen)        │
│  │   ├── lora_A                     │
│  │   └── lora_B                     │
│  └── ...                            │
└─────────────────────────────────────┘
```

### The Math Behind LoRA

For each adapted layer, the forward pass becomes:
```
output = base_layer(x) + (lora_B @ lora_A)(x) * (lora_alpha / r)
```

Where:
- `base_layer(x)` = original frozen weights
- `lora_A`: shape `(r, in_features)` - compresses input
- `lora_B`: shape `(out_features, r)` - expands back
- `lora_alpha / r` = scaling factor

### Merged vs Adapter-Based Models

| Aspect | Adapter-Based (Current) | Merged Model |
|--------|------------------------|--------------|
| Loading | Load base + apply adapter | Load single model |
| Files | Base model + `adapter_model.safetensors` | Single set of model weights |
| Speed | Slower to load | Faster to load |
| Flexibility | Can swap adapters | Fixed weights |
| Deployment | Requires PEFT library | Standard HuggingFace loading |

### To Create a Merged Model (Ready for Deployment)

```python
# After loading with adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained("path/to/merged_model")
processor.save_pretrained("path/to/merged_model")
```

After merging, you can load it directly without PEFT:
```python
model = Qwen2VLForConditionalGeneration.from_pretrained("path/to/merged_model")
```

### Summary

**Current state**: Models are evaluated with adapters loaded on top of base models. They work for inference but require the PEFT library.

**For deployment**: You'd want to run `merge_and_unload()` to combine the adapter weights into the base model, creating a standalone model that's ready for production use.

---

## Understanding `adapter_config.json`

When you save a LoRA adapter with PEFT, it creates an `adapter_config.json` file that stores metadata about the adapter, including which base model it was trained on. Here's what r3-stage2's looks like:

```json
{
  "base_model_name_or_path": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage1",
  "lora_alpha": 128,
  "lora_dropout": 0.1,
  "r": 64,
  "target_modules": ["up_proj", "down_proj", "gate_proj", "v_proj", "o_proj", "q_proj", "k_proj"],
  "peft_type": "LORA",
  ...
}
```

The key field `base_model_name_or_path` tells us which base model the adapter was trained on:

| Adapter | `base_model_name_or_path` | Meaning |
|---------|---------------------------|---------|
| r1-llm-only | `Qwen/Qwen2-VL-7B-Instruct` | Original Qwen base |
| r3-stage2 | `/ssd1/.../r3-stage1` | Stage 1 checkpoint (fine-tuned vision) |
| r4-joint | `Qwen/Qwen2-VL-7B-Instruct` | Original Qwen base |

The updated code reads this field to load the correct base model automatically.

---

## Code Changes

### File Modified
`notebooks/04_evaluation_analysis.ipynb` - Cell `f0bd38d1` (`load_and_evaluate_recipe` function)

### Changes Made

```python
def load_and_evaluate_recipe(recipe_name, model_path, is_lora=True):
    """Load a recipe model and evaluate it."""

    # === FIX 2: AGGRESSIVE GPU MEMORY CLEARING (before loading) ===
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ... (print statements) ...

    if is_lora:
        # === FIX 1: Read adapter_config.json to find the correct base model ===
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            actual_base = adapter_config.get("base_model_name_or_path", BASE_MODEL_ID)
            print(f"  Base model (from adapter_config): {actual_base}")
        else:
            actual_base = BASE_MODEL_ID
            print(f"  Base model (default): {actual_base}")

        # LoRA model: load correct base + adapters
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            actual_base,  # <-- Use actual_base instead of hardcoded BASE_MODEL_ID
            ...
        )
        model = PeftModel.from_pretrained(model, model_path, ...)
```

---

## Actual Results After Fix

| Model | Before Fix | After Fix (Actual) |
|-------|------------|---------------------|
| r3-two-stage | IoU 0.62, 100% det | **IoU 0.8366, 100% det** |
| r4-joint | IoU 0.00, 0% det | **IoU 0.8636, 100% det** |

Both fixes worked as expected. r4-joint is now the best-performing recipe!

---

## Testing Instructions

1. Restart the notebook kernel (fresh GPU state)
2. Run cells 1-9 (setup through failure analysis)
3. Run Section 10 (cell `f0bd38d1` then `ce10917d`)
4. Verify:
   - r3-two-stage output shows: `"Base model (from adapter_config): /ssd1/.../r3-stage1"`
   - r3-two-stage IoU improves to ~0.83
   - r4-joint completes without "q must be on CUDA" errors
   - r4-joint shows reasonable IoU (~0.80+)
