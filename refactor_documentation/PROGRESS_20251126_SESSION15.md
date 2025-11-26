# Session 15 Progress - Fix Buggy Extracted Modules

**Date:** 2025-11-26
**Duration:** ~30 minutes
**Outcome:** Fixed buggy code in extracted `src/` modules

---

## Summary

The extracted `src/` modules were created BEFORE the critical bug fix in Session 11. This session fixed the buggy patterns that would cause trainable params = 0 when used with TRL 0.22+ and SFTTrainer.

---

## The Bug (Background from Session 11)

In TRL 0.22.0.dev0, SFTTrainer internally handles LoRA setup. If you manually call:
1. `prepare_model_for_kbit_training()`
2. `get_peft_model()`

...and then pass the PeftModel to SFTTrainer, it calls `prepare_model_for_kbit_training()` AGAIN, freezing ALL parameters including LoRA adapters (trainable=0).

**Symptoms:**
- Validation loss stays exactly flat (e.g., 3.289622 → 3.289622)
- trainable params: 0 after SFTTrainer creation
- Model learns nothing

**Solution:** Let SFTTrainer handle everything via `peft_config=lora_config`

---

## Changes Made

### 1. Fixed `src/models/loader.py`

**Bug 1 - attn_implementation default (line 47):**
```python
# BEFORE (buggy)
use_flash_attention: bool = True

# AFTER (fixed)
use_flash_attention: bool = False
```

Reason: Training should use `sdpa`, not `flash_attention_2`. The main file uses `sdpa`.

**Bug 2 - LEGACY warning added to `prepare_for_kbit_training()`:**

Added prominent warning in docstring:
```
LEGACY FUNCTION - For TRL < 0.21 only.

⚠️ WARNING: Do NOT use with TRL 0.22+ and SFTTrainer!
SFTTrainer handles this internally when you pass peft_config.
Using this function with SFTTrainer will cause:
    - trainable params = 0 (frozen LoRA)
    - validation loss stays flat
    - model learns nothing

See GitHub issue: https://github.com/huggingface/trl/issues/3926
```

### 2. Fixed `src/models/lora.py`

**LEGACY warning added to `apply_lora()`:**

Added prominent warning in docstring:
```
LEGACY FUNCTION - For TRL < 0.21 only.

⚠️ WARNING: Do NOT use with TRL 0.22+ and SFTTrainer!
Instead, pass lora_config directly to SFTTrainer via peft_config.
Using apply_lora() + SFTTrainer causes trainable params = 0.

CORRECT PATTERN (TRL 0.22+):
    model, processor = load_quantized_model()
    lora_config = create_lora_config()  # Just create config, don't apply!
    trainer = SFTTrainer(
        model=model,  # Pass base model, NOT PeftModel
        peft_config=lora_config,  # Let SFTTrainer handle LoRA
        ...
    )
```

### 3. Fixed `scripts/train.py`

**Had the exact buggy pattern - completely fixed:**

Before (buggy):
```python
model, processor = load_quantized_model(..., use_flash_attention=True)
model = prepare_for_kbit_training(model)  # LEGACY
model = apply_lora(model, lora_config)    # LEGACY
trainer = SFTTrainer(model=model, ...)    # No peft_config - BUG!
```

After (fixed):
```python
model, processor = load_quantized_model(..., use_flash_attention=False)
lora_config = create_lora_config(...)  # Just create, don't apply
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,  # Let SFTTrainer handle LoRA
    ...
)
trainer.model.print_trainable_parameters()  # Verify ~161M params
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/models/loader.py` | Changed `use_flash_attention=False` default + LEGACY warning |
| `src/models/lora.py` | Added LEGACY warning to `apply_lora()` |
| `scripts/train.py` | Fixed buggy pattern - now uses `peft_config` correctly |

---

## Notebooks Status

| Notebook | Status | Notes |
|----------|--------|-------|
| 01_dataset_exploration | ✅ OK | No training code |
| 02_model_understanding | ✅ OK | No training code |
| 03_verify_refactored_modules | ✅ OK | Already rendered |
| 04_evaluation_analysis | ✅ OK | Inference only, `flash_attention_2` is fine |

---

## Correct Usage Pattern (TRL 0.22+)

```python
from src.models.loader import load_quantized_model
from src.models.lora import create_lora_config  # OK to use
from src.training.config import create_sft_config
from trl import SFTTrainer

# Step 1: Load model (DON'T call prepare_for_kbit_training)
model, processor = load_quantized_model()  # Defaults to sdpa

# Step 2: Create LoRA config (DON'T call apply_lora)
lora_config = create_lora_config()

# Step 3: Let SFTTrainer handle everything
trainer = SFTTrainer(
    model=model,
    args=create_sft_config(),
    peft_config=lora_config,  # SFTTrainer handles LoRA internally
    ...
)

# Verify: Should show ~161M trainable params (1.9%)
trainer.model.print_trainable_parameters()
```

---

## What NOT to Do (TRL 0.22+)

```python
# DON'T DO THIS - causes trainable=0 bug
model = prepare_for_kbit_training(model)  # LEGACY
model = apply_lora(model, lora_config)    # LEGACY
trainer = SFTTrainer(model=model, ...)    # PeftModel + no peft_config = BUG
```

---

## Reference

- Session 11: Critical bug discovered and fixed in main file
- Session 13: Same fix applied to trainer_v3
- GitHub Issue: https://github.com/huggingface/trl/issues/3926
