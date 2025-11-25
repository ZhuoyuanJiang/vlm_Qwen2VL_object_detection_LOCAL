"""
Model-related utilities for Qwen2-VL.

This module provides:
- Model loading with quantization (loader.py)
- LoRA configuration and setup (lora.py)
- Inference utilities (inference.py)
"""

from .inference import run_qwen2vl_inference, parse_qwen_bbox_output
from .loader import create_bnb_config, load_quantized_model, prepare_for_kbit_training
from .lora import create_lora_config, apply_lora, DEFAULT_TARGET_MODULES

__all__ = [
    # Inference
    "run_qwen2vl_inference",
    "parse_qwen_bbox_output",
    # Model loading
    "create_bnb_config",
    "load_quantized_model",
    "prepare_for_kbit_training",
    # LoRA
    "create_lora_config",
    "apply_lora",
    "DEFAULT_TARGET_MODULES",
]
