# Session 13 Progress - Fix Second Trainer and Cleanup

**Date:** 2025-11-25

---

## Summary

Applied the same LoRA bug fix from Session 11 to the second trainer (trainer_v3) and cleaned up callback references.

---

## Changes Made

### 1. Fixed trainer_v3 (Lines 2991-3001)

**Problem:** After the Session 11 fix, trainer_v3 was failing with:
```
ValueError: You cannot perform fine-tuning on purely quantized models.
Please attach trainable adapters on top of the quantized model...
```

**Fix:** Added `peft_config=lora_config` to trainer_v3:

```python
# BUG FIX (2025-11-25): Same fix as first trainer - pass peft_config instead of
# assuming model already has LoRA adapters. This prevents the double-preparation bug.
trainer_v3 = SFTTrainer(
    model=model,
    args=training_args_v3,
    train_dataset=train_dataset_formatted,
    eval_dataset=eval_dataset_formatted,
    data_collator=collate_fn_v3,
    processing_class=processor,
    peft_config=lora_config,  # BUG FIX: Let SFTTrainer handle LoRA setup properly
)
```

### 2. Added Debug Print (Lines 3014-3019)

```python
# DEBUG: Check if LoRA parameters are trainable after SFTTrainer creation
print("\n" + "="*60)
print("CHECKING TRAINABLE PARAMETERS AFTER trainer_v3 CREATION")
print("="*60)
trainer_v3.model.print_trainable_parameters()
```

### 3. Commented Out Callback References (Lines 3021-3034)

```python
# =============================================================================
# COMMENTED OUT: Callback usage (requires import from src.training.callbacks)
#
# To re-enable:
#   from src.training.callbacks import GradNormCallback, IoUEvalCallback, ConsoleLogCallback
#
# _iou_cb_v3 = IoUEvalCallback(...)
# trainer_v3.add_callback(_iou_cb_v3)
# _grad_cb_v3 = GradNormCallback()
# trainer_v3.add_callback(_grad_cb_v3)
# trainer_v3.add_callback(ConsoleLogCallback())
# =============================================================================
```

### 4. attn_implementation Check

Verified locations:
- Line 1378: `sdpa` ✅ (main training model)
- Lines 641, 2250, 2543: `flash_attention_2` (testing/inference sections - OK)

**No changes needed** - training already uses `sdpa`.

---

## Files Modified

- `fine_tuning_vlm_for_object_detection_trl.py`
- `fine_tuning_vlm_for_object_detection_trl.ipynb` (synced)
