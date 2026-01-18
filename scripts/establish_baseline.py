#!/usr/bin/env python3
"""
Establish Baseline for Quantization Experiments

This script:
1. Creates a deterministic validation slice (100 samples)
2. Runs BF16 model inference via vLLM server
3. Caches baseline outputs for later comparison with quantized models

Usage:
    # First start vLLM server with BF16 model:
    # CUDA_VISIBLE_DEVICES=0 vllm serve /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
    #   --served-model-name qwen2vl-nutrition --dtype bfloat16 --trust-remote-code \
    #   --max-model-len 4096 --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization 0.9 --port 8000

    # Then run this script:
    python scripts/establish_baseline.py --num-samples 100 --output-dir /ssd1/zhuoyuan/vlm_outputs/quantization_experiments

Environment:
    conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset
from torchvision import ops
import torch
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.inference import parse_qwen_bbox_output


# =============================================================================
# Configuration
# =============================================================================
VLLM_HOST = "localhost"
VLLM_PORT = 8000
MODEL_NAME = "qwen2vl-nutrition"

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
class SampleResult:
    """Result for a single validation sample."""
    sample_idx: int
    ground_truth_bbox: list  # [y_min, x_min, y_max, x_max] in [0,1] (OpenFoodFacts format)
    raw_output: str
    parsed_bbox: Optional[list]  # [x1, y1, x2, y2] in [0,1000) (Qwen format) or None
    iou: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class BaselineResults:
    """Complete baseline evaluation results."""
    model_path: str
    model_type: str  # "bf16", "gptq-int4", etc.
    num_samples: int
    timestamp: str
    metrics: dict
    samples: list  # List of SampleResult as dicts


# =============================================================================
# Utility Functions
# =============================================================================
def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()


def compute_iou(pred_bbox_qwen: list, gt_bbox_off: list) -> float:
    """
    Compute IoU between predicted and ground truth bboxes.

    Args:
        pred_bbox_qwen: [x1, y1, x2, y2] in [0,1000) Qwen format
        gt_bbox_off: [y_min, x_min, y_max, x_max] in [0,1] OpenFoodFacts format

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Normalize predicted bbox from [0,1000) to [0,1]
    pred_norm = [
        pred_bbox_qwen[0] / 1000.0,
        pred_bbox_qwen[1] / 1000.0,
        pred_bbox_qwen[2] / 1000.0,
        pred_bbox_qwen[3] / 1000.0
    ]

    # Convert GT from [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max]
    y_min, x_min, y_max, x_max = gt_bbox_off
    gt_norm = [x_min, y_min, x_max, y_max]

    # Compute IoU using torchvision
    pred_tensor = torch.tensor([pred_norm], dtype=torch.float32)
    gt_tensor = torch.tensor([gt_norm], dtype=torch.float32)

    return ops.box_iou(pred_tensor, gt_tensor).item()


def send_inference_request(image_b64: str, timeout: float = 120.0) -> tuple:
    """
    Send inference request to vLLM server.

    Returns:
        (output_text, latency_ms, error)
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": USER_PROMPT}
            ]}
        ],
        "max_tokens": 64,
        "temperature": 0.0,  # Deterministic
        "skip_special_tokens": False
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"], latency_ms, None
        else:
            return None, latency_ms, str(result)
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return None, latency_ms, str(e)


def check_vllm_server() -> bool:
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# =============================================================================
# Main Functions
# =============================================================================
def create_validation_slice(num_samples: int = 100) -> list:
    """
    Create a deterministic validation slice.

    Returns:
        List of (sample_idx, image, ground_truth_bbox) tuples
    """
    print(f"Loading validation dataset...")
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")

    # Take first N samples deterministically
    num_samples = min(num_samples, len(ds))

    samples = []
    for idx in range(num_samples):
        example = ds[idx]
        image = example['image']
        gt_bbox = example['objects']['bbox'][0]  # First bbox
        samples.append((idx, image, gt_bbox))

    print(f"Created validation slice with {len(samples)} samples")
    return samples


def run_baseline_evaluation(
    samples: list,
    model_path: str,
    model_type: str = "bf16",
) -> BaselineResults:
    """
    Run baseline evaluation on the validation slice.

    Args:
        samples: List of (sample_idx, image, ground_truth_bbox) tuples
        model_path: Path to the model being evaluated
        model_type: Type of model ("bf16", "gptq-int4", etc.)

    Returns:
        BaselineResults with all metrics and per-sample results
    """
    print(f"\nRunning baseline evaluation ({model_type})...")
    print(f"Model: {model_path}")

    results = []
    ious = []
    successful_detections = 0

    for sample_idx, image, gt_bbox in tqdm(samples, desc="Evaluating"):
        # Convert image to base64
        image_b64 = image_to_base64(image)

        # Send request
        raw_output, latency_ms, error = send_inference_request(image_b64)

        if error:
            results.append(SampleResult(
                sample_idx=sample_idx,
                ground_truth_bbox=gt_bbox,
                raw_output="",
                parsed_bbox=None,
                iou=0.0,
                latency_ms=latency_ms,
                success=False,
                error=error
            ))
            ious.append(0.0)
            continue

        # Parse output
        parsed = parse_qwen_bbox_output(raw_output)

        if parsed:
            successful_detections += 1
            # Get bbox
            if isinstance(parsed, list):
                pred_bbox = parsed[0]['bbox']
            else:
                pred_bbox = parsed['bbox']

            # Compute IoU
            iou = compute_iou(pred_bbox, gt_bbox)
            ious.append(iou)

            results.append(SampleResult(
                sample_idx=sample_idx,
                ground_truth_bbox=gt_bbox,
                raw_output=raw_output,
                parsed_bbox=pred_bbox,
                iou=iou,
                latency_ms=latency_ms,
                success=True
            ))
        else:
            results.append(SampleResult(
                sample_idx=sample_idx,
                ground_truth_bbox=gt_bbox,
                raw_output=raw_output,
                parsed_bbox=None,
                iou=0.0,
                latency_ms=latency_ms,
                success=True,  # Request succeeded but no valid bbox
                error="No valid bbox parsed"
            ))
            ious.append(0.0)

    # Compute metrics
    latencies = [r.latency_ms for r in results]
    metrics = {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "max_iou": float(np.max(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "detection_rate": successful_detections / len(samples),
        "iou_threshold_0.5": sum(1 for iou in ious if iou > 0.5) / len(samples),
        "iou_threshold_0.7": sum(1 for iou in ious if iou > 0.7) / len(samples),
        "avg_latency_ms": float(np.mean(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
    }

    return BaselineResults(
        model_path=model_path,
        model_type=model_type,
        num_samples=len(samples),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        samples=[asdict(r) for r in results]
    )


def print_results(results: BaselineResults):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION RESULTS ({results.model_type.upper()})")
    print("=" * 60)
    print(f"Model: {results.model_path}")
    print(f"Samples: {results.num_samples}")
    print(f"Timestamp: {results.timestamp}")

    m = results.metrics
    print(f"\n[Accuracy Metrics]")
    print(f"  Mean IoU:        {m['mean_iou']:.4f}")
    print(f"  Median IoU:      {m['median_iou']:.4f}")
    print(f"  Detection Rate:  {m['detection_rate']:.2%}")
    print(f"  IoU > 0.5:       {m['iou_threshold_0.5']:.2%}")
    print(f"  IoU > 0.7:       {m['iou_threshold_0.7']:.2%}")

    print(f"\n[Latency Metrics]")
    print(f"  Avg Latency:     {m['avg_latency_ms']:.1f} ms")
    print(f"  Min Latency:     {m['min_latency_ms']:.1f} ms")
    print(f"  Max Latency:     {m['max_latency_ms']:.1f} ms")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Establish baseline for quantization experiments")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of validation samples (default: 100)")
    parser.add_argument("--output-dir", type=str,
                       default="/ssd1/zhuoyuan/vlm_outputs/quantization_experiments",
                       help="Output directory for results")
    parser.add_argument("--model-path", type=str,
                       default="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged",
                       help="Path to the model being evaluated")
    parser.add_argument("--model-type", type=str, default="bf16",
                       help="Model type label (bf16, gptq-int4, etc.)")
    parser.add_argument("--skip-server-check", action="store_true",
                       help="Skip vLLM server check")

    args = parser.parse_args()

    # Check vLLM server
    if not args.skip_server_check:
        print("Checking vLLM server...")
        if not check_vllm_server():
            print(f"\nERROR: vLLM server not running at {VLLM_HOST}:{VLLM_PORT}")
            print("\nStart the server with:")
            print(f"  CUDA_VISIBLE_DEVICES=0 vllm serve {args.model_path} \\")
            print(f"    --served-model-name {MODEL_NAME} --dtype bfloat16 --trust-remote-code \\")
            print(f"    --max-model-len 4096 --limit-mm-per-prompt '{{\"image\":1}}' \\")
            print(f"    --gpu-memory-utilization 0.9 --port {VLLM_PORT}")
            sys.exit(1)
        print("vLLM server is running!")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create validation slice
    samples = create_validation_slice(args.num_samples)

    # Save validation slice metadata (for reproducibility)
    slice_metadata = {
        "num_samples": len(samples),
        "sample_indices": [s[0] for s in samples],
        "ground_truth_bboxes": [s[2] for s in samples],
    }
    slice_path = output_dir / "validation_slice_metadata.json"
    with open(slice_path, 'w') as f:
        json.dump(slice_metadata, f, indent=2)
    print(f"Saved validation slice metadata to: {slice_path}")

    # Run evaluation
    results = run_baseline_evaluation(samples, args.model_path, args.model_type)

    # Print results
    print_results(results)

    # Save results
    output_file = output_dir / f"{args.model_type}_baseline_outputs.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
