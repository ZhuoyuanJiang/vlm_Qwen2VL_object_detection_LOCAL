"""
Data utilities for Qwen2-VL fine-tuning.

This module provides:
- Collators for training (collators.py)
- Dataset utilities (dataset.py)
"""

from .collators import (
    # Main collator classes
    AllTokensCollator,
    AssistantOnlyCollator,
    FlashAttentionPatchCollator,  # Legacy
    # Utility functions
    restore_images_in_conversations,
    # Deprecated aliases (kept for backwards compatibility)
    collate_fn_fixed_1,
    collate_fn_fixed_3,
    collate_fn_fixed_fixed1,
)

__all__ = [
    # Main collator classes
    "AllTokensCollator",
    "AssistantOnlyCollator",
    "FlashAttentionPatchCollator",
    # Utility functions
    "restore_images_in_conversations",
    # Deprecated aliases
    "collate_fn_fixed_1",
    "collate_fn_fixed_3",
    "collate_fn_fixed_fixed1",
]
