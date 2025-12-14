#!/usr/bin/env python3
"""
Training script for Qwen2-VL nutrition table detection.

This is a clean training script that uses the extracted modules from src/.
It's much shorter (~150 lines) than the original 4000+ line notebook.

Usage:
    python scripts/train.py

Requirements:
    - GPU with 40GB+ VRAM (or 2x GPUs with 24GB each)
    - wandb account (optional, can disable)
    - TRL 0.21+ (uses peft_config pattern, not manual LoRA)

NOTE (TRL 0.22+ compatibility):
    This script passes peft_config to SFTTrainer and lets it handle LoRA setup.
    Do NOT call prepare_for_kbit_training() or apply_lora() manually - this causes
    trainable params = 0 bug. See: https://github.com/huggingface/trl/issues/3926
"""

# =============================================================================
# STEP 0: GPU Setup (MUST be before importing torch!)
# =============================================================================
from src.utils.gpu import setup_gpus, clear_memory

# Configure GPUs BEFORE importing torch
setup_gpus(use_dual_gpu=True, max_gpus=2)

# Now safe to import torch and other libraries
import torch
from datasets import load_dataset
from trl import SFTTrainer
import wandb

# Import our extracted modules
from src.models.loader import load_quantized_model  # Don't import prepare_for_kbit_training (legacy)
from src.models.lora import create_lora_config  # Don't import apply_lora (legacy)
from src.training.config import create_training_config, print_training_config
from src.data.collators import AllTokensCollator
from src.data.dataset import convert_to_conversation_format, _has_image

# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================
CONFIG = {
    # Model
    "model_id": "Qwen/Qwen2-VL-7B-Instruct",

    # LoRA
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.1,

    # Training
    "output_dir": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora",
    "logging_dir": "/ssd1/zhuoyuan/vlm_outputs/logs",
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "eval_steps": 50,
    "save_steps": 300,

    # Dataset
    "dataset_id": "openfoodfacts/nutrition-table-detection",
    "train_split": "train",
    "eval_split": "test",  # Using test as eval since no val split

    # W&B
    "wandb_project": "qwen2vl-nutrition-detection",
    "wandb_run_name": "qwen2vl-7b-nutrition-lora",
}

# =============================================================================
# STEP 1: Load and Prepare Dataset
# =============================================================================
print("\n" + "="*60)
print("STEP 1: Loading Dataset")
print("="*60)

dataset = load_dataset(CONFIG["dataset_id"])
print(f"Dataset loaded: {CONFIG['dataset_id']}")
print(f"Train samples: {len(dataset[CONFIG['train_split']])}")
print(f"Eval samples: {len(dataset[CONFIG['eval_split']])}")

# Format dataset for training
print("\nFormatting dataset...")
train_dataset = dataset[CONFIG["train_split"]]
eval_dataset = dataset[CONFIG["eval_split"]]

# Apply formatting and filter invalid samples
train_dataset_formatted = train_dataset.map(
    convert_to_conversation_format,
    remove_columns=train_dataset.column_names
).filter(_has_image)

eval_dataset_formatted = eval_dataset.map(
    convert_to_conversation_format,
    remove_columns=eval_dataset.column_names
).filter(_has_image)

print(f"Formatted train samples: {len(train_dataset_formatted)}")
print(f"Formatted eval samples: {len(eval_dataset_formatted)}")

# =============================================================================
# STEP 2: Load Model with Quantization
# =============================================================================
print("\n" + "="*60)
print("STEP 2: Loading Model with 4-bit Quantization")
print("="*60)

# NOTE: use_flash_attention=False (sdpa) is safer for training
# flash_attention_2 can be used for inference only
model, processor = load_quantized_model(
    model_id=CONFIG["model_id"],
    device_map="balanced",
    use_flash_attention=False,  # Use sdpa for training
)

# =============================================================================
# STEP 3: Create LoRA Configuration
# =============================================================================
print("\n" + "="*60)
print("STEP 3: Creating LoRA Configuration")
print("="*60)

# NOTE: For TRL 0.22+, we DON'T call prepare_for_kbit_training() or apply_lora()
# SFTTrainer handles this internally when we pass peft_config
# See: https://github.com/huggingface/trl/issues/3926

lora_config = create_lora_config(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
)
print(f"LoRA config created (will be applied by SFTTrainer)")
print(f"  Rank: {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Dropout: {lora_config.lora_dropout}")

# =============================================================================
# STEP 4: Create Training Configuration
# =============================================================================
print("\n" + "="*60)
print("STEP 4: Creating Training Configuration")
print("="*60)

training_args = create_sft_config(
    output_dir=CONFIG["output_dir"],
    logging_dir=CONFIG["logging_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    eval_steps=CONFIG["eval_steps"],
    save_steps=CONFIG["save_steps"],
    run_name=CONFIG["wandb_run_name"],
)
print_training_config(training_args)

# =============================================================================
# STEP 5: Initialize W&B (Optional)
# =============================================================================
print("\n" + "="*60)
print("STEP 5: Initializing Weights & Biases")
print("="*60)

wandb.login()
wandb.init(
    project=CONFIG["wandb_project"],
    name=CONFIG["wandb_run_name"],
    config={
        "model": CONFIG["model_id"],
        "lora_r": CONFIG["lora_r"],
        "lora_alpha": CONFIG["lora_alpha"],
        "learning_rate": CONFIG["learning_rate"],
        "batch_size": CONFIG["per_device_train_batch_size"],
        "gradient_accumulation": CONFIG["gradient_accumulation_steps"],
        "epochs": CONFIG["num_train_epochs"],
        "dataset": CONFIG["dataset_id"],
    },
    tags=["qwen2-vl", "object-detection", "nutrition-table", "lora"],
)
print(f"✅ W&B initialized: {wandb.run.get_url()}")

# =============================================================================
# STEP 6: Create Data Collator
# =============================================================================
print("\n" + "="*60)
print("STEP 6: Creating Data Collator")
print("="*60)

collate_fn = AllTokensCollator(processor, model)
print("✅ AllTokensCollator created (trains on all non-vision tokens)")

# =============================================================================
# STEP 7: Create Trainer and Train
# =============================================================================
print("\n" + "="*60)
print("STEP 7: Creating SFTTrainer and Starting Training")
print("="*60)

# IMPORTANT: Pass peft_config to let SFTTrainer handle LoRA setup
# DO NOT manually call prepare_for_kbit_training() or apply_lora()
# See: https://github.com/huggingface/trl/issues/3926
trainer = SFTTrainer(
    model=model,  # Base quantized model (NOT a PeftModel)
    args=training_args,
    train_dataset=train_dataset_formatted,
    eval_dataset=eval_dataset_formatted,
    data_collator=collate_fn,
    processing_class=processor,
    peft_config=lora_config,  # Let SFTTrainer handle LoRA setup
)

# CRITICAL: Verify trainable parameters after SFTTrainer creation
# Should show ~161M trainable params (1.9%), NOT 0!
print("\n" + "="*60)
print("VERIFYING TRAINABLE PARAMETERS")
print("="*60)
trainer.model.print_trainable_parameters()
# If this shows 0 trainable params, something is wrong!

print(f"\nTotal training samples: {len(train_dataset_formatted)}")
print(f"Total evaluation samples: {len(eval_dataset_formatted)}")
effective_batch = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
steps_per_epoch = len(train_dataset_formatted) // effective_batch
total_steps = steps_per_epoch * CONFIG["num_train_epochs"]
print(f"Effective batch size: {effective_batch}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}")

print("\n🚀 Starting training...")
trainer.train()

# =============================================================================
# STEP 8: Save Model
# =============================================================================
print("\n" + "="*60)
print("STEP 8: Saving Model")
print("="*60)

trainer.save_model(CONFIG["output_dir"])
processor.save_pretrained(CONFIG["output_dir"])
print(f"✅ Model saved to {CONFIG['output_dir']}")

# Finish W&B run
wandb.finish()
print("\n✅ Training complete!")
