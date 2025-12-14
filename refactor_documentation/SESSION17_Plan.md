# Refactoring Plan: Clean Up & Consolidate Training Configs

**Date:** 2024-12-14
**Goal:** Clean up legacy files, consolidate configs, rename collators for clarity

---

## Context

User plans to add more training recipes:
- v1: All tokens (current)
- v2: Assistant-only (current, was v3)
- v3 (future): Freeze LLM, train encoder only
- v4 (future): Train LLM only

Need a flexible config function and cleaner collator naming.

---

## Tasks Overview

| # | Task | File(s) | Priority |
|---|------|---------|----------|
| 1 | Delete `analyze_tokens.py` | Root directory | High |
| 2 | Create flexible `create_training_config()` | Main file | High |
| 3 | Replace v1 and v3 configs with function calls | Main file | High |
| 4 | Rename collators for clarity | Main file | High |
| 5 | Move `collate_fn_fixed_fixed1` to end as draft | Main file | Medium |
| 6 | Update `src/training/config.py` | src/ | Medium |
| 7 | Update `src/data/collators.py` | src/ | Medium |

**Note:** No helper functions for collators (future consideration - can add later to reduce duplication).

---

## Task 1: Delete `analyze_tokens.py`

**Reason:** Functionality exists in main file's `analyze_token_distribution()`.

**Action:** Delete `/home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/analyze_tokens.py`

---

## Task 2: Create Flexible `create_training_config()` Function

**Location:** Add in main file, before the first `training_args = SFTConfig(...)` block (~line 1537)

**Design:** Use single `variant` parameter for both output path and run name:

```python
def create_training_config(
    # === VARIANT NAME (used for output dir and W&B run name) ===
    variant: str,  # e.g., "v1-all-tokens", "v2-assistant-only", "v3-encoder-only"

    # === HYPERPARAMETERS (with sensible defaults) ===
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,

    # === EVAL/SAVE FREQUENCY ===
    eval_steps: int = 50,
    save_steps: int = 300,

    # === SEQUENCE SETTINGS ===
    max_length: int = 2048,

    # === ADVANCED (rarely changed) ===
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    seed: int = 42,
) -> SFTConfig:
    """
    Create SFT training configuration for any training variant.

    Args:
        variant: Name like "v1-all-tokens", "v2-assistant-only", etc.
                 Used for output_dir suffix and W&B run_name.

    Supports: v1 (all tokens), v2 (assistant-only), v3 (encoder-only), v4 (LLM-only)
    """
    base_output_dir = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection"
    base_logging_dir = "/ssd1/zhuoyuan/vlm_outputs/logs"

    return SFTConfig(
        # Output and logging - use variant name for paths
        output_dir=f"{base_output_dir}-{variant}",
        logging_dir=f"{base_logging_dir}-{variant}",
        logging_steps=10,

        # Training hyperparameters
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,  # DISABLED - causes issues with QLoRA

        # Learning rate and optimization
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        optim="adamw_torch",
        adam_beta2=0.999,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,

        # Mixed precision and performance
        bf16=True,
        tf32=True,
        dataloader_num_workers=0,

        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Other settings
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb",
        run_name=f"qwen2vl-{variant}",
        seed=seed,

        # TRL specific - CONSISTENT for all variants
        dataset_text_field="",
        max_length=max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
```

---

## Task 3: Replace Existing Configs with Function Calls

**v1 config (line ~1537):**
```python
# BEFORE: 55 lines of SFTConfig(...)
# AFTER:
training_args = create_training_config(
    variant="v1-all-tokens",
    eval_steps=50,
    save_steps=300,
)
```

**v2 config (was v3, line ~2885):**
```python
# BEFORE: 48 lines of SFTConfig(...)
# AFTER:
training_args_v2 = create_training_config(
    variant="v2-assistant-only",
    eval_steps=100,
    save_steps=100,
)
```

**Future variants:**
```python
training_args_v3 = create_training_config(
    variant="v3-encoder-only",
    learning_rate=1e-4,  # Maybe different for encoder
)

training_args_v4 = create_training_config(
    variant="v4-llm-only",
    learning_rate=5e-6,  # Maybe different for LLM
)
```

---

## Task 4: Rename Collators for Clarity

**Rename mapping:**

| Current Name | New Name | Description |
|--------------|----------|-------------|
| `collate_fn_fixed_1` | `AllTokensCollator` | Trains on all non-vision tokens |
| `collate_fn_fixed_3` | `AssistantOnlyCollator` | Trains only on assistant responses |
| `collate_fn_fixed_fixed1` | (move to end as draft) | Flash attention workaround |

**Update all usages in main file:**
- `collate_fn = collate_fn_fixed_1(...)` → `collate_fn = AllTokensCollator(...)`
- `collate_fn_v3 = collate_fn_fixed_3(...)` → `collate_fn_v2 = AssistantOnlyCollator(...)`

---

## Task 5: Move `collate_fn_fixed_fixed1` to End as Draft

**Move to end of file with clear documentation:**

```python
# =============================================================================
# DRAFT / LEGACY COLLATORS
# =============================================================================
# The following collators are kept for reference but are NOT used in training.
# =============================================================================

class FlashAttentionPatchCollator:
    """
    LEGACY: Flash Attention dtype workaround.

    This was a workaround for Flash Attention 2 requiring bfloat16 pixel values.
    Since training now uses sdpa (not flash_attention_2), this is NOT needed.

    Kept for reference in case flash_attention_2 is used for inference.
    """
    # ... original collate_fn_fixed_fixed1 code ...
```

---

## Task 6: Update `src/training/config.py`

**Replace two functions with one flexible function:**
- Delete `create_sft_config()`
- Delete `create_assistant_only_training_config()`
- Add new `create_training_config()` (same as main file)

**Keep backwards compatibility (optional):**
```python
# Deprecated aliases
def create_sft_config(**kwargs):
    """Deprecated: Use create_training_config() instead."""
    return create_training_config(variant="v1-all-tokens", **kwargs)
```

---

## Task 7: Update `src/data/collators.py`

**Apply same renames:**
- Rename `collate_fn_fixed_1` → `AllTokensCollator`
- Rename `collate_fn_fixed_3` → `AssistantOnlyCollator`
- Move `collate_fn_fixed_fixed1` to end as legacy
- Update `src/data/__init__.py` exports

---

## Files to Modify

| File | Action |
|------|--------|
| `analyze_tokens.py` | DELETE |
| `fine_tuning_vlm_for_object_detection_trl.py` | Add config function, rename collators |
| `src/training/config.py` | Replace with single flexible function |
| `src/data/collators.py` | Rename collators |
| `src/data/__init__.py` | Update exports |

---

## Future Consideration

Helper functions (like `_restore_images_in_examples()`, `_build_batch_tensors()`) could be added later to reduce code duplication in collators. Currently keeping collators self-contained for readability.

---

## Verification

1. Test that training still works with new config function and renamed collators
2. Verify W&B run names are correct
3. Search for old collator names - should find none (except in legacy section)
