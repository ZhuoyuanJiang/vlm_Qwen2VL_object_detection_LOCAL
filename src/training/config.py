"""
Training configuration for SFTTrainer.

This module provides functions for creating SFT (Supervised Fine-Tuning)
training configurations for Qwen2-VL models.

Extracted from: fine_tuning_vlm_for_object_detection_trl.py

Usage:
    from src.training.config import create_training_config

    training_args = create_training_config(
        variant="v1-all-tokens",
        learning_rate=1e-5,
    )
"""

import warnings
from typing import Optional
from trl import SFTConfig


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
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        per_device_eval_batch_size: Eval batch size per GPU
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate (2e-5 typical for fine-tuning)
        warmup_steps: Linear warmup steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        max_length: Maximum sequence length
        lr_scheduler_type: LR scheduler ("cosine", "linear", "constant")
        weight_decay: Weight decay for regularization
        max_grad_norm: Gradient clipping threshold
        seed: Random seed for reproducibility

    Returns:
        SFTConfig: Configuration object for SFTTrainer

    Supports: v1 (all tokens), v2 (assistant-only), v3 (encoder-only), v4 (LLM-only)

    Example:
        >>> args = create_training_config(
        ...     variant="v1-all-tokens",
        ...     eval_steps=50,
        ...     save_steps=300,
        ... )
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


def print_training_config(config: SFTConfig):
    """
    Print a summary of the training configuration.

    Args:
        config: SFTConfig to summarize
    """
    print("Training Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Number of epochs: {config.num_train_epochs}")
    print(f"  Random seed: {config.seed}")
    print(f"  Output directory: {config.output_dir}")


# =============================================================================
# DEPRECATED FUNCTIONS (for backwards compatibility)
# =============================================================================

def create_sft_config(**kwargs) -> SFTConfig:
    """Deprecated: Use create_training_config(variant='v1-all-tokens') instead."""
    warnings.warn(
        "create_sft_config is deprecated, use create_training_config(variant='v1-all-tokens')",
        DeprecationWarning,
        stacklevel=2
    )
    return create_training_config(variant="v1-all-tokens", **kwargs)


def create_assistant_only_training_config(**kwargs) -> SFTConfig:
    """Deprecated: Use create_training_config(variant='v2-assistant-only') instead."""
    warnings.warn(
        "create_assistant_only_training_config is deprecated, use create_training_config(variant='v2-assistant-only')",
        DeprecationWarning,
        stacklevel=2
    )
    return create_training_config(variant="v2-assistant-only", **kwargs)
