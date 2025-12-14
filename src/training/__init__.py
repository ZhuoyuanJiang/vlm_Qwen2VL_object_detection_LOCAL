"""
Training utilities for Qwen2-VL fine-tuning.

This module provides:
- Training configuration (config.py)
- Training callbacks (callbacks.py)
- Evaluation utilities (evaluation.py)
"""

from .config import (
    create_training_config,
    print_training_config,
    # Deprecated
    create_sft_config,
    create_assistant_only_training_config,
)
from .callbacks import GradNormCallback, IoUEvalCallback, ConsoleLogCallback
from .evaluation import evaluate_model, print_evaluation_results, compare_models

__all__ = [
    # Config
    "create_training_config",
    "print_training_config",
    # Deprecated config functions (kept for backwards compatibility)
    "create_sft_config",
    "create_assistant_only_training_config",
    # Callbacks
    "GradNormCallback",
    "IoUEvalCallback",
    "ConsoleLogCallback",
    # Evaluation
    "evaluate_model",
    "print_evaluation_results",
    "compare_models",
]
