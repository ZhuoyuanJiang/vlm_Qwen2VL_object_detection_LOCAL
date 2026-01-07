"""
HuggingFace Transformers Baseline Benchmark for Qwen2-VL

This script benchmarks HuggingFace Transformers inference performance to compare against vLLM.
Used to generate resume claims like: "Achieved Nx improvement with vLLM over HF Transformers"

Metrics collected:
- Throughput: images/second (equivalent to req/s in vLLM)
- Latency: avg, min, max per request in ms

Usage:
    python scripts/benchmark_hf_baseline.py --batch-size 1 --num-images 20
    python scripts/benchmark_hf_baseline.py --batch-size 8 --num-images 20 --output results.json
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"

SYSTEM_PROMPT = """You are a nutrition label detector. Your task is to identify nutrition tables/panels in food product images.

When you detect a nutrition table, output its location using this exact format:
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates are in [0,1000) range where (0,0) is top-left.
If no nutrition table is found, say "No nutrition table detected."
"""

USER_PROMPT = "Detect the bounding box coordinates for the nutrition facts table in this image."


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class BatchResult:
    """Result from a batch inference."""
    batch_size: int
    batch_latency_ms: float  # Total time for the batch
    per_image_latency_ms: float  # Average per image
    outputs: list[str]


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    config: dict
    summary: dict
    latency_stats: dict
    per_batch_results: list[dict]


# =============================================================================
# Model Loading
# =============================================================================
def load_model_and_processor(model_path: str, device: str = "cuda"):
    """Load Qwen2-VL model and processor."""
    print(f"Loading model from: {model_path}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    )

    processor = Qwen2VLProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,    # ~200k pixels
        max_pixels=1280 * 28 * 28,   # ~1M pixels
        trust_remote_code=True,
    )

    model.eval()
    return model, processor


# =============================================================================
# Inference Functions
# =============================================================================
def run_single_inference(
    model,
    processor,
    image: Image.Image,
    device: str = "cuda",
) -> tuple[str, float]:
    """
    Run inference on a single image.

    Returns:
        tuple: (output_text, latency_ms)
    """
    # Create conversation format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Timed generation
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text, latency_ms


def run_batch_inference(
    model,
    processor,
    images: list[Image.Image],
    device: str = "cuda",
) -> BatchResult:
    """
    Run inference on a batch of images.

    Note: Qwen2-VL doesn't support true batching with different images,
    so we process sequentially and measure total time.
    """
    outputs = []
    total_latency_ms = 0

    for image in images:
        output, latency_ms = run_single_inference(model, processor, image, device)
        outputs.append(output)
        total_latency_ms += latency_ms

    return BatchResult(
        batch_size=len(images),
        batch_latency_ms=total_latency_ms,
        per_image_latency_ms=total_latency_ms / len(images),
        outputs=outputs,
    )


# =============================================================================
# Benchmark Runner
# =============================================================================
def run_benchmark(
    batch_size: int = 1,
    num_images: int = 20,
    warmup_images: int = 3,
    verbose: bool = True,
) -> BenchmarkResults:
    """
    Run HuggingFace Transformers benchmark.

    Args:
        batch_size: Number of images per batch (processed sequentially for HF)
        num_images: Total number of images to process
        warmup_images: Number of warmup iterations
        verbose: Print progress

    Returns:
        BenchmarkResults with all metrics
    """
    if verbose:
        print("=" * 60)
        print(f"HuggingFace Transformers Benchmark")
        print(f"  Batch size: {batch_size}")
        print(f"  Num images: {num_images}")
        print(f"  Warmup: {warmup_images}")
        print("=" * 60)

    # Load model
    if verbose:
        print("\n1. Loading model and processor...")
    model, processor = load_model_and_processor(MODEL_PATH)

    # Load test images
    if verbose:
        print("\n2. Loading test images from dataset...")
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")

    # Use first image for all tests (same as vLLM benchmark)
    test_image = ds[0]['image']
    if verbose:
        print(f"   Image size: {test_image.size}")

    # Warmup
    if verbose:
        print(f"\n3. Running {warmup_images} warmup iterations...")
    for i in range(warmup_images):
        _, latency = run_single_inference(model, processor, test_image)
        if verbose:
            print(f"   Warmup {i+1}: {latency:.1f} ms")

    # Main benchmark
    if verbose:
        print(f"\n4. Running benchmark ({num_images} images, batch_size={batch_size})...")

    num_batches = (num_images + batch_size - 1) // batch_size
    batch_results = []
    all_latencies = []

    torch.cuda.synchronize()
    benchmark_start = time.perf_counter()

    for batch_idx in range(num_batches):
        # Get images for this batch
        batch_images = [test_image] * min(batch_size, num_images - batch_idx * batch_size)

        # Run batch
        result = run_batch_inference(model, processor, batch_images)
        batch_results.append(result)

        # Track individual latencies
        # Since we process sequentially, estimate per-image latency
        for _ in batch_images:
            all_latencies.append(result.per_image_latency_ms)

    torch.cuda.synchronize()
    benchmark_end = time.perf_counter()

    total_time = benchmark_end - benchmark_start
    actual_images = sum(r.batch_size for r in batch_results)
    throughput = actual_images / total_time

    # Compute latency stats
    avg_latency = sum(all_latencies) / len(all_latencies)
    min_latency = min(all_latencies)
    max_latency = max(all_latencies)

    # Build results
    results = BenchmarkResults(
        config={
            "model_path": MODEL_PATH,
            "batch_size": batch_size,
            "num_images": actual_images,
            "warmup_images": warmup_images,
        },
        summary={
            "total_time_s": total_time,
            "throughput_img_per_s": throughput,
            "throughput_req_per_s": throughput,  # Same for single-image requests
        },
        latency_stats={
            "avg_ms": avg_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
        },
        per_batch_results=[
            {
                "batch_size": r.batch_size,
                "batch_latency_ms": r.batch_latency_ms,
                "per_image_latency_ms": r.per_image_latency_ms,
            }
            for r in batch_results
        ],
    )

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\n[Configuration]")
        print(f"  Model: {MODEL_PATH.split('/')[-1]}")
        print(f"  Batch size: {batch_size}")
        print(f"  Images processed: {actual_images}")

        print(f"\n[Summary]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} img/s (req/s)")

        print(f"\n[Latency]")
        print(f"  Avg: {avg_latency:.1f} ms")
        print(f"  Min: {min_latency:.1f} ms")
        print(f"  Max: {max_latency:.1f} ms")

        print("\n" + "=" * 60)

        # Show sample output
        if batch_results and batch_results[0].outputs:
            print(f"\n[Sample Output]")
            print(f"  {batch_results[0].outputs[0][:100]}...")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HuggingFace Transformers inference for Qwen2-VL"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of images per batch (processed sequentially)"
    )
    parser.add_argument(
        "--num-images", type=int, default=20,
        help="Total number of images to process"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    results = run_benchmark(
        batch_size=args.batch_size,
        num_images=args.num_images,
        warmup_images=args.warmup,
        verbose=not args.quiet,
    )

    if args.output:
        output_dict = {
            "config": results.config,
            "summary": results.summary,
            "latency_stats": results.latency_stats,
            "per_batch_results": results.per_batch_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
