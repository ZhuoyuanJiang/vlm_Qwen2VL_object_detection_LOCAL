"""
Model loading utilities for Qwen2-VL with quantization support.

This module provides functions for loading Qwen2-VL models with
4-bit quantization (QLoRA) support.

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 1351-1443)

Usage:
    from src.models.loader import load_quantized_model, prepare_for_kbit_training

    model, processor = load_quantized_model()
    model = prepare_for_kbit_training(model)
"""

from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import prepare_model_for_kbit_training
import torch


def create_bnb_config():
    """
    Create 4-bit quantization config for QLoRA.

    This configuration uses NF4 (Normal Float 4-bit) quantization with
    double quantization for optimal memory efficiency.

    Returns:
        BitsAndBytesConfig: Configuration for 4-bit quantization

    Note:
        - NF4 is optimal for normally distributed weights
        - Double quantization further reduces memory
        - bfloat16 compute dtype provides good precision/speed balance
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_quantized_model(
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    device_map: str = "balanced",
    use_flash_attention: bool = True,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
):
    """
    Load Qwen2-VL model with 4-bit quantization.

    Args:
        model_id: HuggingFace model ID (default: "Qwen/Qwen2-VL-7B-Instruct")
        device_map: Device placement strategy:
            - "balanced": Evenly distribute across available GPUs
            - "auto": Let accelerate decide
            - "sequential": Fill GPUs sequentially
        use_flash_attention: Whether to use Flash Attention 2 (recommended)
        min_pixels: Minimum pixels for image processing (~200k default)
        max_pixels: Maximum pixels for image processing (~1M default)

    Returns:
        Tuple[model, processor]: Loaded model and processor

    Example:
        >>> model, processor = load_quantized_model()
        >>> print(f"Memory: {model.get_memory_footprint() / 1024**3:.2f} GB")
    """
    bnb_config = create_bnb_config()

    # "flash_attention_2" = fastest (requires flash-attn package)
    # "sdpa" = PyTorch's native Scaled Dot-Product Attention (good fallback)
    # "eager" = naive implementation (slowest)
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    processor = Qwen2VLProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    print(f"Model loaded: {model_id}")
    print(f"Model device map: {model.hf_device_map}")
    print(f"Model memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")

    return model, processor


def prepare_for_kbit_training(model, verbose: bool = True):
    """
    Prepare quantized model for k-bit training.

    This function wraps peft's prepare_model_for_kbit_training() with
    optional verbose output showing parameter counts before and after.

    Args:
        model: Quantized model from load_quantized_model()
        verbose: Print parameter summary before/after (default: True)

    Returns:
        Model prepared for training with:
            - Gradient checkpointing hooks removed (if present)
            - Input embeddings cast to float32 for training stability
            - Model configured for PEFT training

    Note:
        For TRL version 0.12.0, this must be called manually before
        applying LoRA. Newer TRL versions (0.21+) do this automatically.

    Example:
        >>> model, processor = load_quantized_model()
        >>> model = prepare_for_kbit_training(model)
        >>> # Now ready for LoRA application
    """
    from src.utils.debug import summarize_trainables

    if verbose:
        print("=== Before prepare_model_for_kbit_training ===")
        summarize_trainables(model)

    model = prepare_model_for_kbit_training(model)

    if verbose:
        print("\n✅ Model prepared for k-bit training")
        print("\n=== After prepare_model_for_kbit_training ===")
        summarize_trainables(model)

    return model
