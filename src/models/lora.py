"""
LoRA/QLoRA configuration and setup for Qwen2-VL.

This module provides functions for creating LoRA configurations and
applying LoRA adapters to Qwen2-VL models.

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 1445-1504)

Usage:
    from src.models.lora import create_lora_config, apply_lora

    lora_config = create_lora_config(r=64, lora_alpha=128)
    model = apply_lora(model, lora_config)
"""

from typing import List, Optional
from peft import LoraConfig, get_peft_model


# Default target modules for Qwen2-VL
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj",     # MLP layers
]


def create_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create LoRA configuration for Qwen2-VL.

    Args:
        r: LoRA rank (higher = more capacity, more memory)
            - 8-16: Light fine-tuning
            - 32-64: Standard fine-tuning
            - 128+: Heavy fine-tuning
        lora_alpha: LoRA alpha scaling factor (typically 2*r)
        lora_dropout: Dropout probability for LoRA layers (0.1 typical)
        target_modules: List of module names to apply LoRA to
            Default: attention (q,k,v,o_proj) and MLP (gate,up,down_proj)
        bias: Bias training strategy ("none", "all", "lora_only")
        task_type: Task type for PEFT ("CAUSAL_LM" for text generation)

    Returns:
        LoraConfig: Configuration object for PEFT

    Example:
        >>> config = create_lora_config(r=32, lora_alpha=64)
        >>> print(f"Rank: {config.r}, Alpha: {config.lora_alpha}")
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES.copy()

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
    )


def apply_lora(model, lora_config: Optional[LoraConfig] = None, verbose: bool = True):
    """
    Apply LoRA adapters to model.

    This function applies LoRA (Low-Rank Adaptation) adapters to the model,
    enabling parameter-efficient fine-tuning.

    Args:
        model: Model prepared for k-bit training (from prepare_for_kbit_training)
        lora_config: LoRA configuration (uses create_lora_config() defaults if None)
        verbose: Print configuration and parameter summary (default: True)

    Returns:
        PeftModel: Model with LoRA adapters attached

    Note:
        For TRL 0.12.0, LoRA must be applied manually using get_peft_model().
        Newer TRL versions can accept peft_config directly in SFTTrainer.

    Example:
        >>> from src.models.loader import load_quantized_model, prepare_for_kbit_training
        >>> from src.models.lora import apply_lora
        >>>
        >>> model, processor = load_quantized_model()
        >>> model = prepare_for_kbit_training(model)
        >>> model = apply_lora(model)  # Uses default config
        >>> # Or with custom config:
        >>> model = apply_lora(model, create_lora_config(r=32))
    """
    from src.utils.debug import summarize_trainables

    if lora_config is None:
        lora_config = create_lora_config()

    if verbose:
        print("LoRA Configuration:")
        print(f"  Rank (r): {lora_config.r}")
        print(f"  Alpha: {lora_config.lora_alpha}")
        print(f"  Dropout: {lora_config.lora_dropout}")
        print(f"  Target modules: {lora_config.target_modules}")

    model = get_peft_model(model, lora_config)

    if verbose:
        print("\n✅ LoRA adapters attached to model")
        # PEFT's method includes full base model in total
        print("\n=== PEFT's parameter count (includes full base model) ===")
        model.print_trainable_parameters()
        print("\n=== Custom count (after LoRA) ===")
        summarize_trainables(model)

    return model
