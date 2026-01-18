#!/usr/bin/env python3
"""
Accuracy Evaluation Script for vLLM Server

This script evaluates model accuracy via the vLLM API endpoint.
It can compare results against a cached baseline to measure quantization drift.

Usage:
    # First start vLLM server (BF16 or quantized)
    # Then run:
    python scripts/evaluate_vllm_accuracy.py \
        --num-samples 100 \
        --output-dir /ssd1/zhuoyuan/vlm_outputs/quantization_experiments \
        --model-type gptq-int4 \
        --compare-baseline  # Optional: compare against BF16 baseline

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

import numpy as np
import requests
import torch
from datasets import load_dataset
from torchvision import ops
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
    """Result for a single evaluation sample."""
    sample_idx: int
    ground_truth_bbox: list
    raw_output: str
    parsed_bbox: Optional[list]
    iou: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    model_type: str
    num_samples: int
    timestamp: str
    metrics: dict
    drift_metrics: Optional[dict]  # Comparison against baseline
    samples: list


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
    """Send inference request to vLLM server."""
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
# Evaluation Functions
# =============================================================================
def load_validation_slice(num_samples: int = 100, metadata_path: Optional[Path] = None):
    """
    Load validation slice.

    If metadata_path exists, uses the exact same samples as baseline.
    Otherwise, creates a new slice from the first N samples.
    """
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")

    if metadata_path and metadata_path.exists():
        print(f"Loading validation slice from metadata: {metadata_path}")
        with open(metadata_path) as f:
            metadata = json.load(f)
        indices = metadata["sample_indices"]
        samples = [(idx, ds[idx]['image'], ds[idx]['objects']['bbox'][0]) for idx in indices]
    else:
        print(f"Creating new validation slice with {num_samples} samples")
        num_samples = min(num_samples, len(ds))
        samples = [(i, ds[i]['image'], ds[i]['objects']['bbox'][0]) for i in range(num_samples)]

    return samples


def run_evaluation(samples: list, model_type: str = "unknown") -> EvaluationResults:
    """Run evaluation on samples and return results."""
    print(f"\nRunning evaluation ({model_type})...")

    results = []
    ious = []
    successful_detections = 0

    for sample_idx, image, gt_bbox in tqdm(samples, desc="Evaluating"):
        image_b64 = image_to_base64(image)
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

        parsed = parse_qwen_bbox_output(raw_output)

        if parsed:
            successful_detections += 1
            if isinstance(parsed, list):
                pred_bbox = parsed[0]['bbox']
            else:
                pred_bbox = parsed['bbox']

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
                success=True,
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

    return EvaluationResults(
        model_type=model_type,
        num_samples=len(samples),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        drift_metrics=None,
        samples=[asdict(r) for r in results]
    )


def compute_drift_metrics(current_results: EvaluationResults, baseline_path: Path) -> dict:
    """
    Compare current results against baseline to compute drift metrics.

    Drift metrics help understand how much accuracy changed due to quantization.
    """
    if not baseline_path.exists():
        print(f"Baseline not found at {baseline_path}, skipping drift computation")
        return None

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_samples = {s["sample_idx"]: s for s in baseline["samples"]}
    current_samples = {s["sample_idx"]: s for s in current_results.samples}

    # Compute per-sample IoU drift
    iou_drifts = []
    exact_matches = 0
    output_matches = 0

    for idx in current_samples:
        if idx in baseline_samples:
            baseline_iou = baseline_samples[idx]["iou"]
            current_iou = current_samples[idx]["iou"]
            iou_drifts.append(current_iou - baseline_iou)

            # Check if outputs are identical
            if current_samples[idx]["raw_output"] == baseline_samples[idx]["raw_output"]:
                output_matches += 1
                exact_matches += 1
            # Check if bboxes are identical (even if formatting differs)
            elif current_samples[idx]["parsed_bbox"] == baseline_samples[idx]["parsed_bbox"]:
                exact_matches += 1

    drift_metrics = {
        "mean_iou_drift": float(np.mean(iou_drifts)) if iou_drifts else 0.0,
        "median_iou_drift": float(np.median(iou_drifts)) if iou_drifts else 0.0,
        "max_iou_improvement": float(np.max(iou_drifts)) if iou_drifts else 0.0,
        "max_iou_degradation": float(np.min(iou_drifts)) if iou_drifts else 0.0,
        "exact_match_rate": exact_matches / len(current_samples) if current_samples else 0.0,
        "output_match_rate": output_matches / len(current_samples) if current_samples else 0.0,
        "baseline_mean_iou": baseline["metrics"]["mean_iou"],
        "current_mean_iou": current_results.metrics["mean_iou"],
        "iou_change_absolute": current_results.metrics["mean_iou"] - baseline["metrics"]["mean_iou"],
        "iou_change_relative_pct": ((current_results.metrics["mean_iou"] - baseline["metrics"]["mean_iou"]) /
                                     max(baseline["metrics"]["mean_iou"], 0.001)) * 100,
    }

    return drift_metrics


def print_results(results: EvaluationResults):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({results.model_type.upper()})")
    print("=" * 60)
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

    if results.drift_metrics:
        d = results.drift_metrics
        print(f"\n[Drift vs Baseline]")
        print(f"  Baseline Mean IoU: {d['baseline_mean_iou']:.4f}")
        print(f"  Current Mean IoU:  {d['current_mean_iou']:.4f}")
        print(f"  IoU Change:        {d['iou_change_absolute']:+.4f} ({d['iou_change_relative_pct']:+.2f}%)")
        print(f"  Exact Match Rate:  {d['exact_match_rate']:.2%}")
        print(f"  Output Match Rate: {d['output_match_rate']:.2%}")

        # Interpret drift
        if abs(d['iou_change_relative_pct']) < 1:
            print(f"  Status: NEGLIGIBLE drift (<1%)")
        elif d['iou_change_relative_pct'] < -5:
            print(f"  Status: SIGNIFICANT degradation (>{abs(d['iou_change_relative_pct']):.1f}%)")
        elif d['iou_change_relative_pct'] > 0:
            print(f"  Status: IMPROVED ({d['iou_change_relative_pct']:.1f}% better)")
        else:
            print(f"  Status: MINOR degradation ({abs(d['iou_change_relative_pct']):.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate vLLM model accuracy")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of validation samples")
    parser.add_argument("--output-dir", type=str,
                       default="/ssd1/zhuoyuan/vlm_outputs/quantization_experiments",
                       help="Output directory for results")
    parser.add_argument("--model-type", type=str, default="unknown",
                       help="Model type label (bf16, gptq-int4, etc.)")
    parser.add_argument("--compare-baseline", action="store_true",
                       help="Compare results against BF16 baseline")
    parser.add_argument("--baseline-file", type=str, default="bf16_baseline_outputs.json",
                       help="Baseline file name for comparison")
    parser.add_argument("--skip-server-check", action="store_true",
                       help="Skip vLLM server check")

    args = parser.parse_args()

    # Check vLLM server
    if not args.skip_server_check:
        print("Checking vLLM server...")
        if not check_vllm_server():
            print(f"\nERROR: vLLM server not running at {VLLM_HOST}:{VLLM_PORT}")
            sys.exit(1)
        print("vLLM server is running!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation slice (use metadata if exists for consistency)
    metadata_path = output_dir / "validation_slice_metadata.json"
    samples = load_validation_slice(args.num_samples, metadata_path)

    # Run evaluation
    results = run_evaluation(samples, args.model_type)

    # Compare against baseline if requested
    if args.compare_baseline:
        baseline_path = output_dir / args.baseline_file
        drift_metrics = compute_drift_metrics(results, baseline_path)
        results.drift_metrics = drift_metrics

    # Print results
    print_results(results)

    # Save results
    output_file = output_dir / f"{args.model_type}_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
