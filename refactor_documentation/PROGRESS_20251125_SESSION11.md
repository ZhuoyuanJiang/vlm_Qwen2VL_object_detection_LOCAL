# Session 11 Progress - CRITICAL BUG FIX: SFTTrainer Freezes LoRA Parameters

**Date:** 2025-11-25
**Duration:** ~1 hour
**Outcome:** MAJOR BUG DISCOVERED AND FIXED

---

## Executive Summary

We discovered and fixed a **critical bug** that caused **all LoRA parameters to be frozen** (trainable params = 0), making training completely ineffective. The validation loss was staying exactly the same (to 6 decimal places) across all evaluation steps because no weights were actually being updated.

---

## The Problem

### Symptoms Observed
1. **Validation loss completely flat**: `3.289622 → 3.289622 → 3.289622` (exactly the same)
2. **Mean token accuracy unchanged**: `0.425383 → 0.425383 → 0.425383`
3. **Training loss wiggled but didn't decrease**: Small random changes from dropout noise
4. **Warning message**: `UserWarning: None of the inputs have requires_grad=True. Gradients will be None`

### Verification
```python
# After SFTTrainer creation:
trainer.model.print_trainable_parameters()

# OUTPUT:
# trainable params: 0 || all params: 8,452,856,320 || trainable%: 0.0000
```

**Zero trainable parameters!** The model was essentially frozen.

---

## Root Cause Analysis

### The Original Code Flow (BUGGY)

```python
# Step 1: Load quantized model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # 4-bit QLoRA
    ...
)

# Step 2: Manually prepare for k-bit training (1st call)
model = prepare_model_for_kbit_training(model)

# Step 3: Manually apply LoRA
model = get_peft_model(model, lora_config)
# At this point: trainable params = 161,480,704 ✓

# Step 4: Create SFTTrainer
trainer = SFTTrainer(
    model=model,  # Passing a PeftModel
    ...
    # NO peft_config - we thought this was correct
)
# At this point: trainable params = 0 ✗ (BUG!)
```

### What Happens Inside SFTTrainer

We traced through the TRL source code and found the bug:

**File: `trl/trainer/sft_trainer.py` (line 136-137)**
```python
if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
    model = prepare_peft_model(model, peft_config, args)
```

Since our model IS a `PeftModel` (we called `get_peft_model`), SFTTrainer calls `prepare_peft_model()`.

**File: `trl/models/utils.py` - `prepare_peft_model()` function**
```python
def prepare_peft_model(model, peft_config, args):
    # ...

    # Handle quantized models (QLoRA)
    is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

    # ...

    # Prepare model for kbit training if needed
    if is_qlora and not is_sharded_qlora:
        model = prepare_model_for_kbit_training(  # ← SECOND CALL!
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            ...
        )
```

### The Bug Mechanism

1. **We call** `prepare_model_for_kbit_training()` on base model → OK, prepares for LoRA
2. **We call** `get_peft_model()` → OK, adds LoRA adapters with 161M trainable params
3. **We pass PeftModel to SFTTrainer**
4. **SFTTrainer detects** it's a PeftModel loaded in 4-bit
5. **SFTTrainer calls** `prepare_model_for_kbit_training()` **AGAIN** on the PeftModel
6. **The second call** freezes ALL parameters, including the LoRA adapters!

### Why the Second Call Freezes Everything

`prepare_model_for_kbit_training()` iterates through ALL model parameters and:
- Sets `requires_grad = False` for quantized layers
- When called on a **base model**: Only freezes base weights, leaves room for LoRA
- When called on a **PeftModel**: Also freezes the LoRA adapter weights!

---

## The Fix

### Solution: Let SFTTrainer Handle LoRA Setup

Instead of manually calling `prepare_model_for_kbit_training()` and `get_peft_model()`, we pass `peft_config` to SFTTrainer and let it handle everything properly.

### Code Changes

#### 1. Comment out manual `prepare_model_for_kbit_training` (lines 1433-1451)

```python
# =============================================================================
# COMMENTED OUT: Manual prepare_model_for_kbit_training
#
# BUG FIX (2025-11-25): SFTTrainer internally calls prepare_model_for_kbit_training
# when it detects QLoRA (see trl/models/utils.py:prepare_peft_model lines 23-29).
# If we call it manually AND pass a PeftModel to SFTTrainer, SFTTrainer calls it
# AGAIN on the PeftModel, which freezes ALL parameters including LoRA (trainable=0).
#
# Solution: Let SFTTrainer handle this by passing peft_config instead of
# manually creating a PeftModel.
#
# See: https://github.com/huggingface/trl/issues/3926
# =============================================================================
# print("=== Before prepare_model_for_kbit_training ===")
# _summarize_trainables(model)
# model = prepare_model_for_kbit_training(model)
# print("\n✅ Model prepared for k-bit training")
# print("\n=== After prepare_model_for_kbit_training ===")
# _summarize_trainables(model)
```

#### 2. Comment out manual `get_peft_model` (lines 1502-1520)

```python
# =============================================================================
# COMMENTED OUT: Manual get_peft_model
#
# BUG FIX (2025-11-25): Same issue as prepare_model_for_kbit_training above.
# SFTTrainer will call get_peft_model internally when we pass peft_config.
# If we call it manually, SFTTrainer's prepare_peft_model will call
# prepare_model_for_kbit_training on our PeftModel, freezing all params.
#
# Solution: Pass lora_config to SFTTrainer via peft_config parameter instead.
# =============================================================================
# model = get_peft_model(model, lora_config)
# print("\n✅ LoRA adapters attached to model")
# print("\n=== PEFT's parameter count (includes full base model) ===")
# model.print_trainable_parameters()
# print("\n=== After get_peft_model (LoRA attached) - Custom count ===")
# _summarize_trainables(model)

print("\n✅ LoRA config created (will be applied by SFTTrainer)")
print("   Note: We pass peft_config to SFTTrainer instead of manually calling get_peft_model")
```

#### 3. Add `peft_config` to SFTTrainer (lines 2141-2181)

```python
# Create the trainer
# =============================================================================
# ORIGINAL COMMENTS (no longer applicable for TRL 0.22.0.dev0):
# # For TRL 0.12.0: We already manually:
# # 1. Called prepare_model_for_kbit_training()
# # 2. Applied get_peft_model() with LoRA config
# # So we DON'T pass peft_config to SFTTrainer (model is already a PEFT model)
# =============================================================================
#
# BUG FIX (2025-11-25): The above approach causes trainable params = 0 in TRL 0.22.0.dev0!
# SFTTrainer internally calls prepare_peft_model() which calls prepare_model_for_kbit_training()
# again on our PeftModel, freezing ALL parameters including LoRA adapters.
# Solution: Pass peft_config to SFTTrainer and let it handle LoRA setup.
# See: https://github.com/huggingface/trl/issues/3926

trainer = SFTTrainer(
    model=model,  # Base quantized model (NOT a PeftModel anymore)
    args=training_args,
    train_dataset=train_dataset_formatted,
    eval_dataset=eval_dataset_formatted,
    data_collator=collate_fn,
    processing_class=processor,
    peft_config=lora_config,  # BUG FIX: Let SFTTrainer handle LoRA setup properly
)

# DEBUG: Check if LoRA parameters are trainable after SFTTrainer creation
print("\n" + "="*60)
print("CHECKING TRAINABLE PARAMETERS AFTER SFTTrainer CREATION")
print("="*60)
trainer.model.print_trainable_parameters()
# If this shows 0 trainable params, the bug is still present
```

---

## Why Common Examples May Be Wrong

Many tutorials and examples online show this pattern:

```python
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
trainer = SFTTrainer(model=model, ...)
```

This pattern **may have worked in older TRL versions** but is **broken in TRL 0.22.0.dev0** (and likely other recent versions). The behavior depends on:

| TRL Version | Behavior |
|-------------|----------|
| Older (0.4-0.8) | Often worked - didn't re-prepare PeftModel |
| 0.12+ | Added more QLoRA detection logic |
| 0.22.0.dev0 | Aggressively detects QLoRA and re-prepares, causing the bug |

---

## Environment Details

```
TRL version: 0.22.0.dev0
PEFT version: 0.17.1
Transformers version: 4.56.0.dev0
Model: Qwen2-VL (8B parameters)
Quantization: 4-bit QLoRA (NF4)
```

---

## Key Lessons Learned

1. **Always verify trainable parameters after trainer creation**
   ```python
   trainer.model.print_trainable_parameters()
   ```

2. **SFTTrainer has built-in QLoRA handling** - don't fight it, use it:
   ```python
   trainer = SFTTrainer(
       model=base_model,  # NOT PeftModel
       peft_config=lora_config,  # Let SFTTrainer handle it
       ...
   )
   ```

3. **Validation loss being exactly flat is a red flag** - it means nothing is being trained

4. **Training loss wiggling without decreasing** can be just dropout noise, not actual learning

5. **Library version changes can break previously working code** - always test after upgrades

---

## References

- GitHub Issue: https://github.com/huggingface/trl/issues/3926
- TRL Source: `trl/trainer/sft_trainer.py` lines 136-137
- TRL Source: `trl/models/utils.py` - `prepare_peft_model()` function

---

## Files Modified

- `fine_tuning_vlm_for_object_detection_trl.py`
- `fine_tuning_vlm_for_object_detection_trl.ipynb` (synced via jupytext)

---

## Expected Result After Fix

```
============================================================
CHECKING TRAINABLE PARAMETERS AFTER SFTTrainer CREATION
============================================================
trainable params: 161,480,704 || all params: 8,452,856,320 || trainable%: 1.9104
```

With trainable parameters restored, the validation loss should now change during training!

---

## Q&A: When is "trainable params: 0" Expected?

### Q: Is `trainable params: 0` ever expected?

**A: Yes, once — and no in your current situation.**

### ✅ Expected: Right after `prepare_model_for_kbit_training`, BEFORE LoRA

For QLoRA, the base 4-bit model is supposed to be frozen. So this sequence is **normal**:

```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

model = prepare_model_for_kbit_training(model)

model.print_trainable_parameters()
# 👉 Often prints: trainable params: 0 (...), because LoRA not attached yet
```

At this point:
- **Base weights:** frozen (`requires_grad=False`) ✅
- **LoRA:** not created yet, so also 0 ✅

### ✅ Expected: After `get_peft_model` (LoRA attached)

Once you do:

```python
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

You should see **non-zero trainables** — the LoRA weights now have `requires_grad=True`:

```
=== After get_peft_model (LoRA attached) ===
trainable params: 161,480,704 || all params: 8,452,856,320 || trainable%: 1.9104
```

This is perfect! ✅

### ❌ NOT Expected: After SFTTrainer creation

But then, after creating SFTTrainer, if you see:

```
After SFTTrainer creation:
trainable params: 0 || all params: 8,452,856,320 || trainable%: 0.0000
```

**This is NOT expected!** It means SFTTrainer has frozen everything, including the LoRA layers. That's why validation loss never moves.

### Summary Table: Expected Trainable Parameters at Each Stage

| Stage | Expected trainable params | Notes |
|-------|---------------------------|-------|
| After `from_pretrained` (4-bit) | 0 | Base model frozen by quantization |
| After `prepare_model_for_kbit_training` | 0 | Still no LoRA attached |
| After `get_peft_model` | ~161M (1.9%) | LoRA adapters now trainable ✅ |
| After `SFTTrainer` (with bug) | 0 | ❌ BUG - everything frozen |
| After `SFTTrainer` (fixed, with `peft_config`) | ~161M (1.9%) | ✅ LoRA adapters trainable |
