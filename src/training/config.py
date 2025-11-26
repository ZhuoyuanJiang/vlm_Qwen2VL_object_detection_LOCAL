"""
Training configuration for SFTTrainer.

This module provides functions for creating SFT (Supervised Fine-Tuning)
training configurations for Qwen2-VL models.

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 1506-1584)

Usage:
    from src.training.config import create_sft_config

    training_args = create_sft_config(
        output_dir="/ssd1/user/outputs",
        learning_rate=1e-5,
    )
"""

from typing import Optional
from trl import SFTConfig


def create_sft_config(
    # Output and logging
    output_dir: str = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora",
    logging_dir: str = "/ssd1/zhuoyuan/vlm_outputs/logs",
    logging_steps: int = 10,
    # Training hyperparameters
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    # Learning rate and optimization
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    # Evaluation and saving
    eval_steps: int = 50,
    save_steps: int = 300,
    save_total_limit: int = 3,
    # Sequence settings
    max_length: int = 2048,
    # Logging
    report_to: str = "wandb",
    run_name: str = "qwen2vl-nutrition-detection",
    seed: int = 42,
) -> SFTConfig:
    """
    Create SFT training configuration.

    This function creates a comprehensive SFTConfig with sensible defaults
    optimized for Qwen2-VL fine-tuning with QLoRA.

    Args:
        output_dir: Directory to save model checkpoints (use SSD for speed!)
        logging_dir: Directory for logs
        logging_steps: Log metrics every N steps
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        per_device_eval_batch_size: Eval batch size per GPU
        gradient_accumulation_steps: Steps to accumulate gradients
            (effective batch = per_device * accumulation * num_gpus)
        learning_rate: Learning rate (2e-5 typical for fine-tuning)
        warmup_steps: Linear warmup steps
        lr_scheduler_type: LR scheduler ("cosine", "linear", "constant")
        weight_decay: Weight decay for regularization
        max_grad_norm: Gradient clipping threshold
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum checkpoints to keep
        max_length: Maximum sequence length
        report_to: Logging backend ("wandb", "tensorboard", "none")
        run_name: Name for the training run
        seed: Random seed for reproducibility

    Returns:
        SFTConfig: Configuration object for SFTTrainer

    Note:
        - gradient_checkpointing is DISABLED (causes issues with QLoRA)
        - bf16 is ENABLED (recommended for modern GPUs)
        - tf32 is ENABLED (for Ampere+ GPUs)
        - dataloader_num_workers=0 (avoids multiprocessing issues)
        - skip_prepare_dataset=True (we use custom data collator)

    Example:
        >>> args = create_sft_config(
        ...     output_dir="/ssd1/my_outputs",
        ...     learning_rate=1e-5,
        ...     num_train_epochs=5,
        ... )
        >>> print(f"LR: {args.learning_rate}, Epochs: {args.num_train_epochs}")
    """
    return SFTConfig(
        # Output and logging
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=logging_steps,

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
        bf16=True,   # Use bfloat16 precision
        tf32=True,   # Enable TF32 on Ampere GPUs
        dataloader_num_workers=0,  # Avoid multiprocessing issues

        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Other settings
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=report_to,
        run_name=run_name,
        seed=seed,

        # Vision model specific settings
        # Empty string because we use custom data_collator, not "messages"
        dataset_text_field="",
        max_length=max_length,
        # We handle data preparation ourselves via custom collator
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


def create_assistant_only_training_config(
    output_dir: str = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora-assistantonly",
    logging_dir: str = "/ssd1/zhuoyuan/vlm_outputs/logs-assistantonly",
    # Training hyperparameters (same defaults as standard config)
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    eval_steps: int = 100,
    save_steps: int = 100,
    run_name: str = "qwen2vl-7b-assistant-only",
    seed: int = 42,
) -> SFTConfig:
    """
    Create SFT training configuration for assistant-only training.

    This configuration is designed for use with collate_fn_fixed_3, which
    only trains on assistant responses (bbox coordinates), ignoring system
    prompts and user messages.

    Args:
        output_dir: Directory to save model checkpoints (different from standard training)
        logging_dir: Directory for logs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        per_device_eval_batch_size: Eval batch size per GPU
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate (2e-5 typical for fine-tuning)
        warmup_steps: Linear warmup steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        run_name: Name for the training run (for W&B)
        seed: Random seed for reproducibility

    Returns:
        SFTConfig: Configuration object for SFTTrainer with assistant-only focus

    Why Assistant-Only Training?
        - Standard training trains on ~100% of tokens (system + user + assistant)
        - ~70% of tokens are static prompts that don't help learning
        - Assistant-only trains on ~30% of tokens (actual bbox coordinates)
        - This leads to:
          - Better convergence on the actual detection task
          - Lower training loss
          - More efficient use of compute resources

    Example:
        >>> from src.training.config import create_assistant_only_training_config
        >>> from src.data.collators import collate_fn_fixed_3
        >>>
        >>> args = create_assistant_only_training_config(
        ...     output_dir="/ssd1/my_outputs/assistant-only",
        ... )
        >>> collator = collate_fn_fixed_3(processor, model)
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     args=args,
        ...     data_collator=collator,  # Use collate_fn_fixed_3
        ...     ...
        ... )
    """
    return SFTConfig(
        # Output and logging - Different directory for comparison
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=10,

        # Training hyperparameters (same as original for fair comparison)
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,  # DISABLED - causes issues with QLoRA

        # Learning rate and optimization (same as original)
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=1.0,

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

        # Advanced options
        remove_unused_columns=False,
        label_names=["labels"],

        # Debugging / reporting
        report_to="wandb",
        run_name=run_name,
        seed=seed,

        # TRL specific
        dataset_text_field=None,  # We handle formatting in collate_fn
        packing=False,
    )
