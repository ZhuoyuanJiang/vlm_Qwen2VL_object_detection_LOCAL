#!/usr/bin/env python3
"""
GPTQ INT4 Quantization Script for Qwen2-VL

This script quantizes the LLM portion of the fine-tuned Qwen2-VL model
using GPTQ INT4, while keeping the vision encoder in BF16 for accuracy.

Usage:
    conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving
    python scripts/quantize_model_gptq.py \
        --model-path /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
        --output-path /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
        --num-calibration-samples 128

Key Features:
- Uses GPTQModel's native Qwen2-VL support
- Excludes vision encoder (model.visual.*) from quantization
- Uses multimodal calibration data (images + text prompts)
- Outputs gptq_marlin compatible format for vLLM serving

After quantization, serve with vLLM:
    CUDA_VISIBLE_DEVICES=0 vllm serve <output-path> \
        --served-model-name qwen2vl-nutrition \
        --quantization gptq_marlin \
        --dtype half \
        --trust-remote-code \
        --max-model-len 4096 \
        --limit-mm-per-prompt '{"image":1}' \
        --gpu-memory-utilization 0.9 \
        --port 8000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def create_calibration_dataset_for_gptqmodel(num_samples: int = 128):
    """
    Create calibration dataset in the format GPTQModel expects for Qwen2-VL.

    GPTQModel's Qwen2-VL handler expects:
    - List of conversation messages with images embedded
    - Format: [{"role": "user", "content": [{"type": "image", "image": <PIL.Image>}, {"type": "text", "text": "..."}]}]

    Returns:
        List of conversation dictionaries
    """
    print(f"Creating calibration dataset with {num_samples} samples...")

    # Load the same dataset used for training
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train")
    num_samples = min(num_samples, len(ds))

    user_prompt = "Detect the bounding box of the nutrition table."

    calibration_data = []
    for i in tqdm(range(num_samples), desc="Preparing calibration data"):
        example = ds[i]
        image = example['image']

        # Format for GPTQModel's Qwen2-VL handler
        # Each item is a list of messages (conversation)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        calibration_data.append(conversation)

    print(f"Created {len(calibration_data)} calibration samples")
    return calibration_data


def quantize_model_gptq(
    model_path: str,
    output_path: str,
    num_calibration_samples: int = 128,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    sym: bool = True,
):
    """
    Quantize Qwen2-VL model using GPTQ INT4.

    Args:
        model_path: Path to the fine-tuned BF16 model
        output_path: Where to save the quantized model
        num_calibration_samples: Number of samples for calibration
        bits: Quantization bits (4 for INT4)
        group_size: Group size for quantization (128 is default)
        desc_act: Use descending activation order (slower but slightly better)
        sym: Use symmetric quantization
    """
    from gptqmodel import GPTQModel, QuantizeConfig

    print("=" * 60)
    print("GPTQ INT4 Quantization for Qwen2-VL")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Calibration samples: {num_calibration_samples}")
    print(f"Bits: {bits}, Group size: {group_size}")
    print("=" * 60)

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {gpu_memory:.1f} GB")

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create calibration dataset in GPTQModel's expected format
    print("\n1. Creating calibration dataset...")
    calibration_data = create_calibration_dataset_for_gptqmodel(num_calibration_samples)

    # Configure quantization
    print("\n2. Configuring quantization...")

    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        format="gptq",
    )

    print(f"Quantization config:")
    print(f"  bits: {bits}")
    print(f"  group_size: {group_size}")
    print(f"  desc_act: {desc_act}")
    print(f"  sym: {sym}")

    # Load model with quantization config
    print("\n3. Loading model...")
    start_time = time.time()

    model = GPTQModel.load(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f} seconds")

    # Quantize with calibration data
    # GPTQModel's Qwen2-VL handler will process the conversations with images
    print("\n4. Quantizing model (this may take a while)...")
    start_time = time.time()

    model.quantize(
        calibration_data,
        batch_size=1,  # Process one sample at a time to avoid OOM
    )

    quantize_time = time.time() - start_time
    print(f"\nQuantization completed in {quantize_time/60:.1f} minutes")

    # Save quantized model
    print("\n5. Saving quantized model...")
    model.save(output_path)

    # Copy processor files from source model
    print("\n6. Copying processor files...")
    from transformers import Qwen2VLProcessor
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    # Verify the quantization config was saved
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "quantization_config" in config:
            print(f"\nQuantization config saved successfully:")
            print(f"  {json.dumps(config['quantization_config'], indent=2)}")
        else:
            print("\nWARNING: quantization_config not found in config.json")

    # Calculate model size
    total_size = 0
    for f in output_dir.glob("*.safetensors"):
        total_size += f.stat().st_size
    total_size_gb = total_size / (1024**3)

    print(f"\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print(f"Model size: {total_size_gb:.2f} GB")
    print(f"Time: {(load_time + quantize_time)/60:.1f} minutes total")
    print("\nTo serve with vLLM:")
    print(f"  CUDA_VISIBLE_DEVICES=0 vllm serve {output_path} \\")
    print(f"    --served-model-name qwen2vl-nutrition \\")
    print(f"    --quantization gptq_marlin \\")
    print(f"    --dtype half \\")
    print(f"    --trust-remote-code \\")
    print(f"    --max-model-len 4096 \\")
    print(f"    --limit-mm-per-prompt '{{\"image\":1}}' \\")
    print(f"    --gpu-memory-utilization 0.9 \\")
    print(f"    --port 8000")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GPTQ INT4 quantization for Qwen2-VL")
    parser.add_argument("--model-path", type=str,
                       default="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged",
                       help="Path to the fine-tuned BF16 model")
    parser.add_argument("--output-path", type=str,
                       default="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4",
                       help="Path to save the quantized model")
    parser.add_argument("--num-calibration-samples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--bits", type=int, default=4,
                       help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for quantization (default: 128)")
    parser.add_argument("--desc-act", action="store_true",
                       help="Use descending activation order (slower but may be more accurate)")
    parser.add_argument("--no-sym", action="store_true",
                       help="Use asymmetric quantization (default: symmetric)")

    args = parser.parse_args()

    quantize_model_gptq(
        model_path=args.model_path,
        output_path=args.output_path,
        num_calibration_samples=args.num_calibration_samples,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=not args.no_sym,
    )


if __name__ == "__main__":
    main()
