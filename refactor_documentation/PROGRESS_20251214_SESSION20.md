# Session 20: Debugging r3-two-stage Training Recipe

**Date:** 2024-12-15
**Focus:** Fix dtype mismatch error in two-stage training recipe

---

## Problem Statement

When running the two-stage training recipe:
```bash
python scripts/train_recipe.py --recipe r3-two-stage --gpu 6,7
```

**Stage 1 completed successfully**, but **Stage 2 failed** with:
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

The error occurred at `logits = self.lm_head(hidden_states)` in `modeling_qwen2_vl.py:1404`.

---

## Investigation Process

### Step 1: Analyze the Log File

From `r3-two-stage_20251215_033647.log`, we found:

1. **Stage 1 (r2-vision-only):** Trained successfully for 3 epochs
   - Model loaded in bf16 (no quantization)
   - Vision encoder fully trained (675M parameters)
   - Gradient checkpointing enabled
   - Saved to `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage1`

2. **Stage 2 (r1-llm-only from checkpoint):** Failed immediately
   - Tried to load Stage 1 checkpoint with 4-bit quantization
   - Error at first training step

### Step 2: Understand the Two-Stage Recipe Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    r3-two-stage Recipe                          │
├─────────────────────────────────────────────────────────────────┤
│ Stage 1: r2-vision-only                                         │
│   - Load: Original HuggingFace model (bf16)                     │
│   - Train: Vision encoder (full fine-tuning)                    │
│   - Freeze: LLM                                                 │
│   - Save: Full model to checkpoint                              │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: r1-llm-only (from Stage 1 checkpoint)                  │
│   - Load: Stage 1 checkpoint (with 4-bit quantization)          │
│   - Train: LLM (LoRA)                                           │
│   - Freeze: Vision encoder                                      │
│   - Save: Final model                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Step 3: Root Cause Analysis (Initial Hypothesis - Later Disproven)

> **Note:** This section documents our initial hypothesis based on the error message. It was later **disproven** by notebook experiments. See "Session 20 Continued" for the actual root cause.

#### The Error Breakdown:
- `mat1` (hidden_states) = **float32**
- `mat2` (lm_head.weight) = **bfloat16**

#### Why This Happens:

1. **Stage 1 saves model in bf16 format** (standard after full fine-tuning)
   - Gradient checkpointing was enabled for memory efficiency
   - Some parameters may have float32 artifacts from gradient checkpointing

2. **Stage 2 loads with 4-bit quantization on-the-fly**
   - BitsAndBytes quantizes LLM linear layers to 4-bit
   - Vision encoder is NOT quantized (not in target modules)
   - Vision encoder retains its original dtype (mixed float32/bf16)

3. **During forward pass:**
   - Vision encoder outputs float32 hidden states
   - These flow through the LLM
   - `lm_head` (in bf16) receives float32 input → **dtype mismatch!**

#### Key Insight: "Load in 4-bit" Explained

This was a source of confusion during debugging. Clarification:

```
"Load in 4-bit" does NOT mean the checkpoint is stored in 4-bit format.
It means: load weights from checkpoint (bf16) and quantize on-the-fly to 4-bit.

Checkpoint file (safetensors)     Loading Process          GPU Memory
┌─────────────────────────┐      ┌──────────────┐      ┌─────────────────┐
│ Weights stored as bf16  │  →   │ Quantize to  │  →   │ Weights in 4-bit│
│ (standard format)       │      │ 4-bit on fly │      │ (nf4 format)    │
└─────────────────────────┘      └──────────────┘      └─────────────────┘
```

---

## Options Considered

### Option A: Skip Quantization in Stage 2

**Approach:** Load Stage 1 checkpoint in bf16 (no quantization), apply LoRA on bf16 model.

| Pros | Cons |
|------|------|
| Simple fix (add one parameter) | Different memory footprint than r1 |
| Guaranteed dtype consistency | Unfair comparison (bf16 vs 4-bit) |
| bf16 has better numerical precision | Uses ~25-30 GB instead of ~16-20 GB |

### Option B: Keep Quantization + Cast Vision Encoder to bf16

**Approach:** Load with 4-bit quantization, then explicitly cast vision encoder parameters to bf16.

| Pros | Cons |
|------|------|
| Fair comparison to r1-llm-only | Slightly more code |
| Same memory footprint (~16-20 GB) | Need to understand model structure |
| Maintains original design intent | |

---

## Decision: Option B (Based on Wrong Hypothesis)

> **Note:** This decision was based on the incorrect parameter dtype hypothesis. The actual fix turned out to be different - see "Session 20 Continued" section.

### Rationale:

1. **Fair Comparison is Critical**
   - The purpose of r3 is to test: "Does pre-training the vision encoder help?"
   - For fair comparison, r3-Stage2 should use the same setup as r1-llm-only
   - Both should use 4-bit quantization + LoRA

2. **Hardware Consideration**
   - Available: 2x RTX 6000 Ada (48 GB each) = 96 GB total
   - Either option fits in VRAM
   - But keeping quantization maintains consistency with other recipes

3. **Minimal Change**
   - Only modifies `load_model_for_recipe()` function
   - Doesn't change training logic or recipe definitions
   - Easy to understand and maintain

---

## First Attempted Fix (Did Not Work)

### Hypothesis: Vision encoder parameters were float32

We initially hypothesized that the vision encoder parameters were being saved/loaded as float32, causing the dtype mismatch.

### Code Change Attempted:

```python
# Added after model loading:
if checkpoint_path and recipe.quantized:
    cast_count = 0
    for name, param in model_recipe.named_parameters():
        if ".visual." in name and param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            cast_count += 1
    if cast_count > 0:
        print(f"  Cast {cast_count} vision encoder parameters to bf16")
```

### Result: **Did not fix the issue**

The notebook showed that **no float32 parameters existed** - all vision encoder params were already bf16 or uint8. The fix cast 0 parameters and the error persisted.

**This fix was reverted** since it had no effect.

---

## Current Investigation: DataParallel Hypothesis (CONFIRMED ✓)

> **Update:** This hypothesis was **confirmed** to be the actual root cause. See "Session 20 Continued" for the fix.

### The Mystery

After the notebook experiments, we observed something puzzling:

| Scenario | Source | Works? |
|----------|--------|--------|
| r1-llm-only (standalone) | HuggingFace | Yes |
| r3-Stage2 | Stage 1 checkpoint | No |

Both use identical settings:
- Same recipe config (r1-llm-only)
- Same 4-bit quantization
- Same `device_map="balanced"`
- Same 2 GPUs

**The ONLY difference is where the model weights are loaded from.**

### Error Traceback Clues

The error traceback shows `torch/nn/parallel/data_parallel.py` is involved:
```
torch/nn/parallel/data_parallel.py:186, in forward
peft/peft_model.py:1850, in forward
```

**We never explicitly used DataParallel in our code.** So where is it coming from?

### Hypothesis (Unverified)

HuggingFace Trainer automatically wraps models with DataParallel when it detects multiple GPUs:
```python
# Inside HuggingFace Trainer (not our code):
if torch.cuda.device_count() > 1 and not model_is_already_distributed:
    model = torch.nn.DataParallel(model)
```

**Our guess:** When loading from HuggingFace, `from_pretrained` might set up something (perhaps `model.hf_device_map`) that tells Trainer "this model is already distributed across devices, don't add DataParallel."

When loading from a custom checkpoint, this setup might not happen correctly, causing Trainer to wrap the model with DataParallel on top of a model that's already distributed via `device_map="balanced"`.

This would create a conflict:
- `device_map="balanced"` = Model parallelism (layers split across GPUs)
- DataParallel = Data parallelism (full model replicated on each GPU)

These two approaches are incompatible, which could explain the dtype mismatch error during the forward pass.

### Next Steps (Resolved)

~~Further debugging needed to understand:~~
1. ~~Why does r1-llm-only work but r3-Stage2 fail when both use the same settings?~~
2. ~~What exactly causes DataParallel to be involved? (We never explicitly used it)~~
3. ~~What is different about loading from HuggingFace vs loading from a custom checkpoint?~~

**Status: RESOLVED** - See "Session 20 Continued" section for the fix.

---

## Recipe Comparison Summary

After this fix, all recipes will work correctly:

| Recipe | Quantization | Vision Training | LLM Training | VRAM |
|--------|--------------|-----------------|--------------|------|
| r1-llm-only | 4-bit | frozen | LoRA | ~16-20 GB |
| r2-vision-only | bf16 | full fine-tune | frozen | ~25-30 GB |
| r3-two-stage | Stage1: bf16, Stage2: 4-bit | Stage1: full | Stage2: LoRA | ~25-30 GB peak |
| r4-joint | 4-bit | LoRA | LoRA | ~16-20 GB |

---

## Debug Notebook Results (05_debug_dtype_issue.ipynb)

We created a notebook to test the dtype hypothesis. **Results disproved our initial hypothesis!**

> **Note:** Part 2 (Tests 7-9) was added later to investigate the DataParallel hypothesis, which turned out to be the actual cause.

| Test | Vision Encoder Dtype | LLM Dtype |
|------|---------------------|-----------|
| HF Original (bf16) | bf16: 391 | bf16: 337 |
| HF Original (4-bit) | bf16: 261, uint8: 130 | uint8: 196, bf16: 141 |
| Stage1 Checkpoint (bf16) | bf16: 391 | bf16: 337 |
| Stage1 Checkpoint (4-bit) | bf16: 261, uint8: 130 | uint8: 196, bf16: 141 |
| After our fix | **Cast 0 params** (no float32!) | - |

**Key findings:**
- Vision encoder parameters are **NOT float32** - they're bf16/uint8
- Our fix cast **0 parameters** because there were none to cast
- Forward pass in inference mode **succeeded**

**The real issue** might be related to:
- Multi-GPU + DataParallel interaction (error traceback shows `data_parallel.py`)
- PEFT/LoRA wrapping during training
- Training mode vs inference mode differences

Our fix is still **defensive** - it won't hurt anything, but the actual root cause may be elsewhere.

---

## VRAM Usage (Stage 1 Training)

During Stage 1 (r2-vision-only, bf16, full vision fine-tuning):
- GPU 6: 13,823 MiB / 49,140 MiB (~28%)
- GPU 7: 12,233 MiB / 49,140 MiB (~25%)
- **Total: ~26 GB** for 2-GPU training

This is reasonable for a 7B model in bf16 with activations and optimizer states.

---

## Lessons Learned (From Initial Investigation)

> **Note:** These lessons are from Part 1, before we found the actual root cause.

1. **Always verify hypotheses with experiments**
   - We assumed float32 parameters based on the error message
   - The notebook showed all parameters were bf16/uint8
   - The actual issue turned out to be DataParallel wrapping (see "Session 20 Continued")

2. **"Load in 4-bit" is on-the-fly quantization**
   - Checkpoints are always saved in standard formats (bf16/fp32)
   - Quantization happens during the loading process
   - Both HF original and custom checkpoints load identically

3. **Inference vs Training can behave differently**
   - Forward pass in inference mode (single GPU, no PEFT) works
   - The error only occurs during training with multi-GPU + PEFT

4. **Fair comparison requires consistent configurations**
   - When comparing recipes, keep as many variables constant as possible
   - Quantization level affects memory but also numerical precision

---

## Testing (Initial - Based on Wrong Hypothesis)

> **Note:** This testing plan was based on the dtype casting fix, which did not work. See "Session 20 Continued" for the correct fix and verification.

```bash
# Test the fix (this approach did NOT work)
python scripts/train_recipe.py --recipe r3-two-stage --gpu 6,7

# Expected output (based on wrong hypothesis):
# Stage 1: Trains vision encoder (bf16)
# Stage 2: Loads checkpoint, casts vision to bf16, trains LLM with LoRA
```

---

## Files Modified (Initial - Reverted)

> **Note:** This change was reverted as it didn't fix the issue.

- ~~`scripts/train_recipe.py` - Added dtype casting for vision encoder when loading checkpoint with quantization~~

---

## Session 20 Continued: Successful Fix (2024-12-15)

### Discovery: The Real Root Cause

After further investigation, we discovered that the dtype mismatch error was **NOT** caused by parameter dtypes (our initial hypothesis), but by **DataParallel wrapping**.

**Key Evidence:**
| Observation | Stage 1 | Stage 2 |
|-------------|---------|---------|
| Progress bar | `0/136` steps | `0/68` steps (half!) |
| DataParallel in traceback | No | **Yes** |
| Training status | Completes | Fails at step 0 |

The step count halving (136 → 68) proves that Trainer wrapped Stage 2 with DataParallel, which:
1. Uses worker threads that don't inherit AMP/autocast context
2. Causes fp32 activations to flow to bf16 `lm_head`
3. Results in the dtype mismatch error

### Why DataParallel Was Applied to Stage 2

HuggingFace Trainer prevents DataParallel when it detects model parallelism via:
- `model.hf_device_map` containing multiple devices
- `model.is_parallelizable=True` and `model.model_parallel=True`

When loading from HuggingFace with `device_map`, these attributes are set correctly.
When loading from a checkpoint, something prevents proper detection, causing Trainer to apply DataParallel on top of the already device_map-distributed model.

### Failed Attempt: Model Parallelism Flags

We first tried setting `model.is_parallelizable = True` and `model.model_parallel = True` after loading the model. **This did not work** because SFTTrainer internally wraps the model with PEFT, and these attributes are lost during the wrapping.

### Successful Fix: Force `_n_gpu=1` for Checkpoint Loading

The working fix sets `training_args._n_gpu = 1` **conditionally** (only when loading from checkpoint):

```python
# In scripts/train_recipe.py, before creating SFTTrainer:
if checkpoint_path and torch.cuda.device_count() > 1:
    training_args_recipe._n_gpu = 1
    print("  [Checkpoint fix] Set _n_gpu=1 to prevent DataParallel (device_map parallelism preserved)")
```

**Why this works:**
- Setting `_n_gpu=1` tells Trainer "don't use DataParallel"
- The model is **still distributed** via `device_map="balanced"` (model parallelism)
- This fix is conditional - only applies when `checkpoint_path` is provided
- Other recipes (r1, r2, r4) that load from HuggingFace are **unaffected**

### Verification

```bash
python scripts/train_recipe.py --recipe r3-two-stage --gpu 6,7 --epochs 1
```

**Results:**
- Stage 1: 136 steps completed successfully (vision encoder training, bf16)
- Stage 2: 136 steps completed successfully (LLM LoRA training from checkpoint)
- No dtype mismatch errors

### Files Modified (Final)

| File | Change |
|------|--------|
| `scripts/train_recipe.py` | Added conditional `_n_gpu=1` fix for checkpoint loading (~line 441) |
| `notebooks/05_debug_dtype_issue.ipynb` | Added Part 2: DataParallel Investigation (Tests 7-9) + Conclusion |

**Debug Notebook Part 2 Contents:**
- **Test 7**: Compare model attributes between HuggingFace vs Checkpoint loading
- **Test 8**: Check device_map details to verify model parallelism detection
- **Test 9**: Check if PEFT wrapping preserves `hf_device_map` attribute
- **Conclusion**: Documents root cause and fix

### Key Lessons

1. **Parameter dtypes ≠ Activation dtypes**: The error message says "float != bfloat16" but the float32 came from activations, not parameters.

2. **DataParallel + device_map are incompatible**: You cannot use both - device_map splits the model across GPUs, DataParallel tries to replicate it.

3. **Setting `_n_gpu=1` doesn't reduce GPU utilization**: The model is still distributed via `device_map="balanced"`. This setting just prevents incompatible DataParallel wrapping.

4. **Conditional fixes are safer**: By only applying the fix when loading from checkpoint, we avoid affecting working recipes.
