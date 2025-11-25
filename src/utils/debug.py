"""
Debugging and diagnostic utilities for model training.

This module provides helper functions for debugging and diagnostics
during model training and development.

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 1402-1431)
"""


def summarize_trainables(model):
    """
    Print summary of trainable parameters in the model.

    This function provides detailed information about:
    - Total vs trainable parameter counts
    - Percentage of trainable parameters
    - Device map (for multi-GPU setups)
    - Sample of trainable tensor names and shapes

    Args:
        model: PyTorch model to summarize

    Example:
        >>> from src.utils.debug import summarize_trainables
        >>> summarize_trainables(model)
        Trainable parameter summary:
          Total params: 7,615,616,000
          Trainable params: 83,886,080 (1.10%)
          Device map snapshot (first 10):
            model.embed_tokens -> cuda:0
            ...
    """
    try:
        total_params, trainable_params = 0, 0
        trainable_examples = []

        for param_name, param_tensor in model.named_parameters():
            param_count = param_tensor.numel()
            total_params += param_count
            if param_tensor.requires_grad:
                trainable_params += param_count
                if len(trainable_examples) < 8:
                    device_str = str(param_tensor.device)
                    trainable_examples.append(
                        f"- {param_name} | shape={tuple(param_tensor.shape)} | device={device_str}"
                    )

        trainable_percentage = 100.0 * trainable_params / max(total_params, 1)

        print("\nTrainable parameter summary:")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({trainable_percentage:.2f}%)")

        if hasattr(model, 'hf_device_map'):
            print("  Device map snapshot (first 10):")
            for i, (module_name, device_location) in enumerate(model.hf_device_map.items()):
                if i >= 10:
                    break
                print(f"    {module_name} -> {device_location}")

        if trainable_examples:
            print("  Sample trainable tensors:")
            for example in trainable_examples:
                print("    ", example)

    except Exception as error:
        print(f"[warn] could not summarize trainables: {error}")
