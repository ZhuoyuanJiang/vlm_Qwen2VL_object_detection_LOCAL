#!/usr/bin/env python3
"""
Merge LoRA adapter weights into base Qwen2-VL model.

This script takes a trained LoRA adapter and merges it into the base model,
creating a standalone model ready for deployment without PEFT dependencies.

Usage:
    python scripts/merge_lora.py

    # Or with custom paths:
    python scripts/merge_lora.py --adapter-path /path/to/adapter --output-path /path/to/output
"""

import argparse
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel


def merge_lora_to_base(
    base_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    adapter_path: str = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint",
    output_path: str = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged",
    device_map: str = "auto",
):
    """
    Merge LoRA adapter into base model and save.

    Args:
        base_model_id: HuggingFace model ID for the base model
        adapter_path: Path to the trained LoRA adapter
        output_path: Path to save the merged model
        device_map: Device placement strategy
    """
    print("=" * 60)
    print("LoRA Merge Script")
    print("=" * 60)
    print(f"Base model: {base_model_id}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    print("=" * 60)

    # Verify adapter exists
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    print("\n1. Loading base model (full precision for merging)...")
    # Load base model in bfloat16 (NOT quantized) for merging
    # We need full-precision weights to merge the LoRA adapters into
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    print(f"   Base model loaded: {base_model.get_memory_footprint() / 1024**3:.2f} GB")

    print("\n2. Loading LoRA adapter...")
    # Load the adapter on top of base model
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print("   Adapter loaded successfully")

    print("\n3. Merging LoRA weights into base model...")
    # Merge the LoRA weights into the base model
    # This adds (alpha/r) * B @ A to the original weights
    merged_model = model.merge_and_unload()
    print("   Merge complete!")
    print(f"   Merged model memory: {merged_model.get_memory_footprint() / 1024**3:.2f} GB")

    print("\n4. Loading processor...")
    processor = Qwen2VLProcessor.from_pretrained(
        base_model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )

    print(f"\n5. Saving merged model to {output_path}...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    print(f"Merged model saved to: {output_path}")
    print("\nFiles created:")
    for f in sorted(output_path.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")

    return merged_model, processor


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Base model ID from HuggingFace",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged",
        help="Path to save merged model",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map strategy (auto, balanced, sequential)",
    )

    args = parser.parse_args()

    merge_lora_to_base(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
