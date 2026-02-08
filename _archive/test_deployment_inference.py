#!/usr/bin/env python3
"""
Test deployed Qwen2-VL model inference via vLLM or Triton endpoints.

DEPRECATED (2026-02-06):
    This was an early deployment smoke-test script committed during the initial
    vLLM/Triton deployment work (late Dec 2025). It is not part of the current,
    actively-maintained evaluation/benchmarking flow and may be deleted later.

    Prefer these scripts instead:
      - vLLM: `scripts/quick_test_vllm_api.py`, `scripts/quick_test_vllm_with_visualization.py`
      - vLLM accuracy: `scripts/evaluate_vllm_accuracy.py`, `scripts/establish_baseline.py`
      - Triton accuracy: `scripts/validate_triton_accuracy.py`
      - Triton benchmarking: `scripts/benchmark_triton.py`

    Note:
      - The `--use-dataset` path in this script uses a `split="test"` dataset
        which does not exist for `openfoodfacts/nutrition-table-detection`
        (it uses `train`/`val`).

This script tests the deployed model by sending inference requests
and verifying the responses contain valid bounding box outputs.

Usage:
    # Test vLLM endpoint
    python scripts/test_deployment_inference.py --endpoint vllm

    # Test Triton endpoint
    python scripts/test_deployment_inference.py --endpoint triton

    # Test with custom image
    python scripts/test_deployment_inference.py --image /path/to/image.jpg

    # Test with image from dataset
    python scripts/test_deployment_inference.py --use-dataset

Prerequisites:
    - vLLM server running (python scripts/serve_vllm.py)
    - OR Triton server running (see scripts/setup_triton.py)
"""

import argparse
import base64
import json
import re
import sys
import requests
from pathlib import Path
from typing import Optional


# Default configuration
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_TRITON_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "qwen2vl-nutrition"
TRITON_MODEL_NAME = "qwen2vl_nutrition"

# Prompts (same as training)
SYSTEM_PROMPT = """You are a nutrition label detector. Your task is to identify nutrition tables/panels in food product images.

When you detect a nutrition table, output its location using this exact format:
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates are in [0,1000) range where (0,0) is top-left.
If no nutrition table is found, say "No nutrition table detected."
"""

USER_PROMPT = "Detect the bounding box coordinates for the nutrition facts table in this image."


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_bbox_output(text: str) -> Optional[dict]:
    """Parse model output to extract bounding box coordinates.

    Expected format:
        <|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

    Returns:
        dict with 'object' and 'bbox' keys, or None if parsing fails
    """
    # Pattern with special tokens
    pattern_with_tokens = r'<\|object_ref_start\|>(.+?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'

    # Pattern without special tokens (fallback)
    pattern_simple = r'(\w+[\w\s-]*)\((\d+),(\d+)\),\((\d+),(\d+)\)'

    # Try with tokens first
    match = re.search(pattern_with_tokens, text)
    if match:
        return {
            'object': match.group(1),
            'bbox': [int(match.group(2)), int(match.group(3)),
                     int(match.group(4)), int(match.group(5))]
        }

    # Try simple pattern
    match = re.search(pattern_simple, text)
    if match:
        return {
            'object': match.group(1).strip(),
            'bbox': [int(match.group(2)), int(match.group(3)),
                     int(match.group(4)), int(match.group(5))]
        }

    return None


def test_vllm_endpoint(
    image_path: str,
    base_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """Test vLLM OpenAI-compatible endpoint.

    Args:
        image_path: Path to test image
        base_url: vLLM server URL
        model_name: Model name configured in vLLM

    Returns:
        dict with 'success', 'output', 'parsed_bbox', 'error' keys
    """
    print(f"\nTesting vLLM endpoint: {base_url}")
    print(f"Model: {model_name}")
    print(f"Image: {image_path}")

    # Check health
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            return {"success": False, "error": f"Health check failed: {resp.status_code}"}
        print("Health check: OK")
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Cannot connect to server: {e}"}

    # Encode image
    try:
        image_b64 = encode_image_to_base64(image_path)
    except Exception as e:
        return {"success": False, "error": f"Cannot read image: {e}"}

    # Build request
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    # Send request
    print("Sending inference request...")
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Inference request failed: {e}"}

    # Parse response
    try:
        result = resp.json()
        output = result["choices"][0]["message"]["content"]
    except (KeyError, json.JSONDecodeError) as e:
        return {"success": False, "error": f"Cannot parse response: {e}"}

    # Parse bounding box
    parsed = parse_bbox_output(output)

    return {
        "success": True,
        "output": output,
        "parsed_bbox": parsed,
        "error": None,
    }


def test_triton_endpoint(
    image_path: str,
    base_url: str = DEFAULT_TRITON_URL,
    model_name: str = TRITON_MODEL_NAME,
) -> dict:
    """Test Triton inference endpoint.

    Args:
        image_path: Path to test image
        base_url: Triton server URL
        model_name: Model name in Triton

    Returns:
        dict with 'success', 'output', 'parsed_bbox', 'error' keys
    """
    print(f"\nTesting Triton endpoint: {base_url}")
    print(f"Model: {model_name}")
    print(f"Image: {image_path}")

    # Check health
    try:
        resp = requests.get(f"{base_url}/v2/health/live", timeout=5)
        if resp.status_code != 200:
            return {"success": False, "error": f"Health check failed: {resp.status_code}"}
        print("Health check: OK")
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Cannot connect to server: {e}"}

    # Encode image
    try:
        image_b64 = encode_image_to_base64(image_path)
    except Exception as e:
        return {"success": False, "error": f"Cannot read image: {e}"}

    # Build full prompt (Triton vLLM backend takes raw text, not chat format)
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {USER_PROMPT}"

    # Build Triton request
    payload = {
        "inputs": [
            {
                "name": "text_input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [full_prompt],
            },
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_b64],
            },
            {
                "name": "sampling_parameters",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps({"max_tokens": 256, "temperature": 0.0})],
            },
        ]
    }

    # Send request
    print("Sending inference request...")
    try:
        resp = requests.post(
            f"{base_url}/v2/models/{model_name}/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Inference request failed: {e}"}

    # Parse response
    try:
        result = resp.json()
        output = result["outputs"][0]["data"][0]
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        return {"success": False, "error": f"Cannot parse response: {e}"}

    # Parse bounding box
    parsed = parse_bbox_output(output)

    return {
        "success": True,
        "output": output,
        "parsed_bbox": parsed,
        "error": None,
    }


def get_sample_image_from_dataset() -> Optional[str]:
    """Load a sample image from the dataset and save to temp file."""
    try:
        from datasets import load_dataset
        from PIL import Image
        import tempfile

        print("Loading sample image from dataset...")
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="test")

        # Get first image
        sample = ds[0]
        image = sample["image"]

        # Save to temp file
        temp_path = "/tmp/test_nutrition_image.jpg"
        image.save(temp_path)

        print(f"Saved sample image to: {temp_path}")
        return temp_path

    except Exception as e:
        print(f"Cannot load from dataset: {e}")
        return None


def main():
    print(
        "WARNING: `scripts/test_deployment_inference.py` is deprecated. "
        "Prefer `scripts/quick_test_vllm_api.py`, `scripts/quick_test_vllm_with_visualization.py`, "
        "`scripts/evaluate_vllm_accuracy.py`, `scripts/benchmark_triton.py`, or "
        "`scripts/validate_triton_accuracy.py`."
    )

    parser = argparse.ArgumentParser(
        description="Test deployed Qwen2-VL model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--endpoint", "-e",
        type=str,
        default="vllm",
        choices=["vllm", "triton", "both"],
        help="Which endpoint to test (default: vllm)"
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to test image"
    )

    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Load test image from HuggingFace dataset"
    )

    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_URL,
        help=f"vLLM server URL (default: {DEFAULT_VLLM_URL})"
    )

    parser.add_argument(
        "--triton-url",
        type=str,
        default=DEFAULT_TRITON_URL,
        help=f"Triton server URL (default: {DEFAULT_TRITON_URL})"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"vLLM model name (default: {DEFAULT_MODEL_NAME})"
    )

    args = parser.parse_args()

    # Get test image
    image_path = args.image
    if image_path is None:
        if args.use_dataset:
            image_path = get_sample_image_from_dataset()
        else:
            # Check for existing test image
            default_paths = [
                "/tmp/test_nutrition_image.jpg",
                "/tmp/test_nutrition.jpg",
            ]
            for p in default_paths:
                if Path(p).exists():
                    image_path = p
                    break

    if image_path is None:
        print("Error: No test image provided")
        print("Use --image /path/to/image.jpg or --use-dataset")
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("Deployment Inference Test")
    print("=" * 60)

    results = {}

    # Test vLLM
    if args.endpoint in ["vllm", "both"]:
        result = test_vllm_endpoint(
            image_path=image_path,
            base_url=args.vllm_url,
            model_name=args.model_name,
        )
        results["vllm"] = result

        print("\n" + "-" * 40)
        print("vLLM Result:")
        print("-" * 40)
        if result["success"]:
            print(f"Output: {result['output']}")
            if result["parsed_bbox"]:
                print(f"Parsed: {result['parsed_bbox']}")
            else:
                print("Warning: Could not parse bounding box from output")
        else:
            print(f"Error: {result['error']}")

    # Test Triton
    if args.endpoint in ["triton", "both"]:
        result = test_triton_endpoint(
            image_path=image_path,
            base_url=args.triton_url,
            model_name=TRITON_MODEL_NAME,
        )
        results["triton"] = result

        print("\n" + "-" * 40)
        print("Triton Result:")
        print("-" * 40)
        if result["success"]:
            print(f"Output: {result['output']}")
            if result["parsed_bbox"]:
                print(f"Parsed: {result['parsed_bbox']}")
            else:
                print("Warning: Could not parse bounding box from output")
        else:
            print(f"Error: {result['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for endpoint, result in results.items():
        status = "PASS" if result["success"] and result.get("parsed_bbox") else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{endpoint.upper()}: {status}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
