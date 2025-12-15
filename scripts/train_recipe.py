#!/usr/bin/env python3
"""
Training script for running recipes in parallel on different GPUs.

This script enables running multiple training recipes simultaneously on different
GPUs. Each recipe can be run as a separate process.

Usage:
    # Run r1-llm-only on GPUs 0,1
    python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1

    # Run r2-vision-only on GPUs 2,3
    python scripts/train_recipe.py --recipe r2-vision-only --gpu 2,3

    # Run in parallel (different terminals):
    # Terminal 1: python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1
    # Terminal 2: python scripts/train_recipe.py --recipe r2-vision-only --gpu 2,3

    # Run two-stage recipe (sequential: r2 -> r1)
    python scripts/train_recipe.py --recipe r3-two-stage --gpu 0,1

Recipes:
    r1-llm-only    : LLM-only QLoRA (4-bit), vision frozen
    r2-vision-only : Vision encoder full fine-tuning, LLM frozen (bf16)
    r3-two-stage   : Two-stage: Vision first -> LLM from checkpoint
    r4-joint       : Joint: Vision LoRA + LLM LoRA together

Requirements:
    - 2 GPUs per training run (model parallelism via device_map="balanced")
    - TRL 0.22+ (uses peft_config pattern)
    - wandb account (optional, use --no-wandb to disable)
"""

# =============================================================================
# SECTION 1: ARGUMENT PARSING (BEFORE ANY TORCH IMPORTS!)
# =============================================================================
import os
import sys
import argparse

# Parse args BEFORE importing torch (GPU setup must be first!)
parser = argparse.ArgumentParser(
    description="Train Qwen2-VL with different recipes",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1
  python scripts/train_recipe.py --recipe r2-vision-only --gpu 2,3
  python scripts/train_recipe.py --recipe r3-two-stage --gpu 0,1
  python scripts/train_recipe.py --recipe r4-joint --gpu 0,1 --no-wandb
"""
)
parser.add_argument(
    "--recipe",
    type=str,
    required=True,
    choices=["r1-llm-only", "r2-vision-only", "r3-two-stage", "r4-joint"],
    help="Recipe to run (required)"
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
    help="GPU ID(s) to use (e.g., '0,1' or '2,3'). If not specified, uses auto-detection."
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs (default: 3)"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint (for loading from previous stage)"
)
parser.add_argument(
    "--model-id",
    type=str,
    default="Qwen/Qwen2-VL-7B-Instruct",
    help="HuggingFace model ID (default: Qwen/Qwen2-VL-7B-Instruct)"
)
parser.add_argument(
    "--no-wandb",
    action="store_true",
    help="Disable W&B logging"
)

args = parser.parse_args()

# =============================================================================
# SECTION 2: GPU SETUP (MUST BE BEFORE IMPORTING TORCH!)
# =============================================================================

# Priority: --gpu flag > CUDA_VISIBLE_DEVICES env var > auto-detection
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPUs specified via --gpu: {args.gpu}")
elif "CUDA_VISIBLE_DEVICES" in os.environ:
    print(f"Using GPUs from CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    # Auto-detect available GPUs
    # Add parent directory to path for src/ imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.gpu import setup_gpus
    setup_gpus(use_dual_gpu=True, max_gpus=2)

# Set memory management for better GPU utilization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================================
# SECTION 3: IMPORTS (NOW SAFE TO IMPORT TORCH!)
# =============================================================================

import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, List
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig
from trl import SFTTrainer

# Add parent directory to path for src/ imports (if not already added)
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collators import AssistantOnlyCollator
from src.training.config import create_training_config
from src.data.dataset import convert_to_conversation_format, _has_image

# Optional W&B import
if not args.no_wandb:
    import wandb

# =============================================================================
# SECTION 4: RECIPE DEFINITIONS
# =============================================================================

# --- Target Module Constants ---
LLM_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # MLP (SwiGLU)
]

VISION_LORA_TARGETS = [
    "qkv",   # Combined Q,K,V attention
    "fc1",   # MLP first layer
    "fc2",   # MLP second layer
    "attn.proj",   # Attention projection
    # NOTE: Omitting "proj" to avoid matching "o_proj" in LLM
]


@dataclass
class TrainingRecipe:
    """Configuration for a training recipe."""
    name: str
    quantized: bool              # True = 4-bit, False = bf16
    train_vision: str            # "none" | "full" | "lora"
    train_llm: str               # "none" | "full" | "lora"
    lora_target_modules: Optional[List[str]]
    learning_rate: float
    description: str


RECIPES = {
    "r1-llm-only": TrainingRecipe(
        name="r1-llm-only",
        quantized=True,
        train_vision="none",
        train_llm="lora",
        lora_target_modules=LLM_LORA_TARGETS,
        learning_rate=1e-4,
        description="Baseline: LLM-only QLoRA (4-bit), vision frozen",
    ),
    "r2-vision-only": TrainingRecipe(
        name="r2-vision-only",
        quantized=False,  # bf16 for full fine-tuning
        train_vision="full",
        train_llm="none",
        lora_target_modules=None,
        learning_rate=5e-5,
        description="Vision encoder full fine-tuning, LLM frozen",
    ),
    "r3-two-stage": TrainingRecipe(
        name="r3-two-stage",
        quantized=True,  # Stage 2 uses 4-bit
        train_vision="none",  # Stage 1 handled separately
        train_llm="lora",
        lora_target_modules=LLM_LORA_TARGETS,
        learning_rate=1e-4,
        description="Two-stage: Vision first (r2) -> LLM (r1) from checkpoint",
    ),
    "r4-joint": TrainingRecipe(
        name="r4-joint",
        quantized=True,
        train_vision="lora",
        train_llm="lora",
        lora_target_modules=LLM_LORA_TARGETS + VISION_LORA_TARGETS,
        learning_rate=1e-4,
        description="Joint: Vision LoRA + LLM LoRA together",
    ),
}

# =============================================================================
# SECTION 5: HELPER FUNCTIONS
# =============================================================================

def load_model_for_recipe(model_id: str, recipe: TrainingRecipe, checkpoint_path: str = None):
    """
    Load model configured for a specific recipe.

    Args:
        model_id: HuggingFace model ID
        recipe: TrainingRecipe configuration
        checkpoint_path: Optional path to load from (for two-stage)

    Returns:
        (model, processor) tuple
    """
    bnb_config_recipe = None
    if recipe.quantized:
        bnb_config_recipe = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    load_path = checkpoint_path if checkpoint_path else model_id

    print(f"\nLoading model from: {load_path}")
    print(f"  Quantization: {'4-bit' if recipe.quantized else 'bf16'}")

    model_recipe = Qwen2VLForConditionalGeneration.from_pretrained(
        load_path,
        quantization_config=bnb_config_recipe,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    processor_recipe = Qwen2VLProcessor.from_pretrained(
        model_id,  # Always use original for processor
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
        trust_remote_code=True,
    )

    model_recipe.config.use_cache = False

    # Enable gradient checkpointing for r2 (vision-only) to reduce memory
    # This is safe for bf16 but NOT for QLoRA recipes
    if recipe.name == "r2-vision-only":
        print("  Enabling gradient checkpointing for r2 (memory optimization)")
        model_recipe.gradient_checkpointing_enable()

    return model_recipe, processor_recipe


def apply_freeze_policy(model, recipe: TrainingRecipe):
    """
    Apply freeze/unfreeze based on recipe configuration.

    Parameter paths from named_parameters():
        Vision: model.visual.blocks.0.attn.qkv.weight
        LLM:    model.language_model.layers.0.self_attn.q_proj.weight
    """
    # Start with everything frozen
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on recipe
    if recipe.train_vision == "full":
        vision_count = 0
        for name, param in model.named_parameters():
            if ".visual." in name:
                param.requires_grad = True
                vision_count += 1
        print(f"  Unfroze {vision_count} vision parameters")

    if recipe.train_llm == "full":
        llm_count = 0
        for name, param in model.named_parameters():
            # Note: "model.layers" would NOT match "model.language_model.layers"!
            if "language_model" in name or "lm_head" in name:
                param.requires_grad = True
                llm_count += 1
        print(f"  Unfroze {llm_count} LLM parameters")

    # For LoRA cases, PEFT will handle unfreezing adapter weights


def build_lora_config_for_recipe(recipe: TrainingRecipe) -> Optional[LoraConfig]:
    """Build LoRA config for a recipe, or None if not using LoRA."""
    if recipe.train_llm != "lora" and recipe.train_vision != "lora":
        return None

    return LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recipe.lora_target_modules,
    )


def load_dataset_for_training():
    """
    Load and prepare the dataset for training.

    Returns:
        (train_dataset_formatted, eval_dataset_formatted)
    """
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)

    dataset_id = "openfoodfacts/nutrition-table-detection"
    dataset = load_dataset(dataset_id)

    print(f"Dataset: {dataset_id}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['val'])}")

    # Format and filter
    train_dataset = dataset["train"]
    eval_dataset = dataset["val"]

    train_dataset_formatted = train_dataset.map(
        convert_to_conversation_format,
        remove_columns=train_dataset.column_names
    ).filter(_has_image)

    eval_dataset_formatted = eval_dataset.map(
        convert_to_conversation_format,
        remove_columns=eval_dataset.column_names
    ).filter(_has_image)

    print(f"Formatted train: {len(train_dataset_formatted)}")
    print(f"Formatted eval: {len(eval_dataset_formatted)}")

    return train_dataset_formatted, eval_dataset_formatted


def run_recipe(
    recipe_key: str,
    model_id: str,
    checkpoint_path: str = None,
    num_epochs: int = 3,
    use_wandb: bool = True,
    train_dataset=None,
    eval_dataset=None,
    output_suffix: str = None,
):
    """
    Run a complete training recipe.

    Args:
        recipe_key: Key from RECIPES dict (e.g., "r1-llm-only")
        model_id: HuggingFace model ID
        checkpoint_path: Optional path to load from (for two-stage)
        num_epochs: Number of training epochs
        use_wandb: Whether to use W&B logging
        train_dataset: Pre-loaded train dataset (loaded if None)
        eval_dataset: Pre-loaded eval dataset (loaded if None)
        output_suffix: Override output directory suffix (e.g., "r3-stage1")

    Returns:
        Output directory path
    """
    recipe = RECIPES[recipe_key]

    # Determine variant name for output directory
    variant_name = output_suffix if output_suffix else recipe.name

    print("\n" + "="*80)
    print(f"RUNNING RECIPE: {recipe.name}")
    print(f"Description: {recipe.description}")
    print(f"Output suffix: {variant_name}")
    print("="*80)

    # Load dataset if not provided
    if train_dataset is None or eval_dataset is None:
        train_dataset, eval_dataset = load_dataset_for_training()

    # Load model
    model_recipe, processor_recipe = load_model_for_recipe(model_id, recipe, checkpoint_path)

    # Apply freeze policy
    print("\nApplying freeze policy...")
    apply_freeze_policy(model_recipe, recipe)

    # Build LoRA config
    lora_config_recipe = build_lora_config_for_recipe(recipe)
    if lora_config_recipe:
        print(f"LoRA config: r={lora_config_recipe.r}, alpha={lora_config_recipe.lora_alpha}")
        print(f"  Target modules: {lora_config_recipe.target_modules}")
    else:
        print("LoRA: Not using (full fine-tuning)")

    # Create training config (use variant_name for output path)
    training_args_recipe = create_training_config(
        variant=variant_name,
        learning_rate=recipe.learning_rate,
        num_train_epochs=num_epochs,
    )

    # Initialize W&B
    if use_wandb:
        wandb.login()
        wandb.init(
            project="qwen2vl-nutrition-detection",
            name=f"qwen2vl-{variant_name}",
            config={
                "model": model_id,
                "recipe": recipe.name,
                "variant": variant_name,
                "learning_rate": recipe.learning_rate,
                "epochs": num_epochs,
                "quantized": recipe.quantized,
                "train_vision": recipe.train_vision,
                "train_llm": recipe.train_llm,
            },
            tags=["qwen2-vl", "recipe", variant_name],
        )
        print(f"W&B: {wandb.run.get_url()}")

    # Create collator
    collator_recipe = AssistantOnlyCollator(processor_recipe, model_recipe)
    collator_recipe.debug = False

    # Create trainer
    trainer_recipe = SFTTrainer(
        model=model_recipe,
        args=training_args_recipe,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_recipe,
        processing_class=processor_recipe,
        peft_config=lora_config_recipe,
    )

    print("\nTrainable parameters after trainer creation:")
    if hasattr(trainer_recipe.model, "print_trainable_parameters"):
        trainer_recipe.model.print_trainable_parameters()
    else:
        # `print_trainable_parameters()` is a PEFT helper method available on PEFT-wrapped models (e.g. `PeftModel`,
        # used when LoRA is enabled). For non-PEFT runs like `r2-vision-only` (plain `Qwen2VLForConditionalGeneration`),
        # the method doesn't exist, so we fall back to a simple trainable/total parameter summary.
        model_to_report = trainer_recipe.model
        trainable_parameter_count = model_to_report.num_parameters(only_trainable=True)
        total_parameter_count = model_to_report.num_parameters()
        trainable_parameter_percent = (
            100.0 * trainable_parameter_count / total_parameter_count
            if total_parameter_count
            else 0.0
        )
        print(
            f"  trainable params: {trainable_parameter_count:,} / {total_parameter_count:,} "
            f"({trainable_parameter_percent:.4f}%)"
        )

    # Calculate training info
    effective_batch = training_args_recipe.per_device_train_batch_size * training_args_recipe.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * num_epochs
    print(f"\nTraining info:")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")

    # Train
    print("\nStarting training...")
    trainer_recipe.train()

    # Save
    trainer_recipe.save_model(training_args_recipe.output_dir)
    processor_recipe.save_pretrained(training_args_recipe.output_dir)

    if use_wandb:
        wandb.finish()

    print(f"\nRecipe {recipe.name} complete! Saved to: {training_args_recipe.output_dir}")

    return training_args_recipe.output_dir


def run_two_stage_recipe(model_id: str, num_epochs: int = 3, use_wandb: bool = True):
    """
    Run the two-stage recipe (r3):
    Stage 1: Vision-only training (r2) -> saves to -r3-stage1
    Stage 2: LLM-only training from Stage 1 checkpoint (r1) -> saves to -r3-stage2

    NOTE: Uses separate output paths (-r3-stage1, -r3-stage2) to avoid
    overwriting standalone r1-llm-only and r2-vision-only outputs.
    """
    print("\n" + "="*80)
    print("TWO-STAGE RECIPE (r3)")
    print("Stage 1: Vision encoder training -> r3-stage1")
    print("Stage 2: LLM training from checkpoint -> r3-stage2")
    print("="*80)

    # Load dataset once for both stages
    train_dataset, eval_dataset = load_dataset_for_training()

    # Stage 1: Vision-only (uses r3-stage1 output path)
    print("\n" + "="*60)
    print("STAGE 1: Vision Encoder Training")
    print("="*60)
    stage1_dir = run_recipe(
        "r2-vision-only",
        model_id,
        num_epochs=num_epochs,
        use_wandb=use_wandb,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_suffix="r3-stage1",  # Override output path
    )

    # Stage 2: LLM-only from Stage 1 checkpoint (uses r3-stage2 output path)
    print("\n" + "="*60)
    print("STAGE 2: LLM Training from Stage 1 Checkpoint")
    print("="*60)
    stage2_dir = run_recipe(
        "r1-llm-only",
        model_id,
        checkpoint_path=stage1_dir,
        num_epochs=num_epochs,
        use_wandb=use_wandb,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_suffix="r3-stage2",  # Override output path
    )

    return stage2_dir


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

def main():
    """Main entry point for recipe training."""

    print("="*60)
    print("RECIPE TRAINING SCRIPT")
    print("="*60)
    print(f"Recipe: {args.recipe}")
    print(f"Model: {args.model_id}")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint: {args.checkpoint or 'None'}")
    print(f"W&B: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Auto-detected')}")

    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"\nPyTorch sees {torch.cuda.device_count()} GPU(s):")
        for i in range(torch.cuda.device_count()):
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GB)")
    else:
        print("\nCUDA not available!")
        sys.exit(1)

    # Run the recipe
    if args.recipe == "r3-two-stage" and args.checkpoint is None:
        # Run both stages sequentially
        output_dir = run_two_stage_recipe(
            model_id=args.model_id,
            num_epochs=args.epochs,
            use_wandb=not args.no_wandb,
        )
    else:
        # Run single recipe
        output_dir = run_recipe(
            recipe_key=args.recipe,
            model_id=args.model_id,
            checkpoint_path=args.checkpoint,
            num_epochs=args.epochs,
            use_wandb=not args.no_wandb,
        )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Output: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
