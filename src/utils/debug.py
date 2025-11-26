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


def analyze_token_distribution(collate_fn, dataset, processor, num_samples=3):
    """
    Analyze what tokens are being trained on vs masked.

    This function helps understand why loss might not be decreasing by showing
    the breakdown of tokens that are actually used for training vs masked.

    Args:
        collate_fn: The collator function/class to test
        dataset: The formatted dataset to sample from
        processor: Qwen2VLProcessor instance (for tokenizer.decode)
        num_samples: Number of samples to analyze (default: 3)

    Example:
        >>> from src.utils.debug import analyze_token_distribution
        >>> from src.data.collators import collate_fn_fixed_1
        >>> collator = collate_fn_fixed_1(processor, model)
        >>> analyze_token_distribution(collator, train_dataset_formatted, processor, num_samples=3)

    Output shows:
        - Total tokens per sample
        - Masked (ignored) tokens count
        - Trained tokens count and percentage
        - Decoded content of what we're training on
    """
    import torch

    print("\n" + "="*60)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Analyzing {num_samples} samples to understand training tokens...\n")

    for i in range(min(num_samples, len(dataset))):
        test_batch = [dataset[i]]
        batch_output = collate_fn(test_batch)

        input_ids = batch_output['input_ids'][0]
        labels = batch_output['labels'][0]

        # Count token types
        total_tokens = len(input_ids)
        masked_tokens = (labels == -100).sum().item()
        trained_tokens = total_tokens - masked_tokens

        print(f"Sample {i+1}:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Masked (ignored): {masked_tokens} tokens")
        print(f"  Trained on: {trained_tokens} tokens ({100*trained_tokens/total_tokens:.1f}%)")

        # Decode what we're training on
        trained_indices = (labels != -100).nonzero(as_tuple=True)[0]
        if len(trained_indices) > 0:
            # Show first and last parts of what we're training on
            trained_token_ids = input_ids[trained_indices]
            decoded = processor.tokenizer.decode(trained_token_ids, skip_special_tokens=False)
            print(f"  Training content: {decoded}")

        print()

    print("📊 BREAKDOWN OF TRAINED TOKENS:")
    print("  ~50 tokens: System prompt (static - same every sample)")
    print("  ~10 tokens: User prompt 'Detect the bounding box...' (static)")
    print("  ~10 tokens: Role markers like <|im_start|>, <|im_end|> (static)")
    print("  ~30 tokens: Assistant response with bbox coordinates (ONLY THIS VARIES!)")
    print()
    print("⚠️  ISSUE: 70% of trained tokens are static (don't help learning)")
    print("    Only ~30% are the actual variable bbox coordinates")
    print("    This is why loss decreases very slowly!")
    print("="*60)
