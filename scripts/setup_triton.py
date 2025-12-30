#!/usr/bin/env python3
"""
Set up Nvidia Triton model repository for Qwen2-VL deployment.

This script creates the necessary configuration files for deploying the
Qwen2-VL model with Triton Inference Server using the vLLM backend.

Usage:
    # Basic usage (uses defaults)
    python scripts/setup_triton.py

    # Custom paths
    python scripts/setup_triton.py \
        --model /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
        --repo /ssd1/zhuoyuan/triton_model_repository

After running this script, launch Triton with:
    docker run --gpus all --rm -it \\
        --shm-size=4G \\
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
        -v /ssd1/zhuoyuan/vlm_outputs:/ssd1/zhuoyuan/vlm_outputs:ro \\
        -v /ssd1/zhuoyuan/triton_model_repository:/models \\
        nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \\
        tritonserver --model-repository=/models
"""

import argparse
import json
import sys
from pathlib import Path


# Default configuration
DEFAULT_MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
DEFAULT_REPO_PATH = "/ssd1/zhuoyuan/triton_model_repository"
DEFAULT_MODEL_NAME = "qwen2vl_nutrition"


def create_model_json(
    model_path: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 4096,
) -> dict:
    """Create model.json configuration for vLLM backend."""
    return {
        "model": model_path,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
    }


def create_config_pbtxt(model_name: str) -> str:
    """Create config.pbtxt for Triton."""
    return f'''name: "{model_name}"
backend: "vllm"

# Maximum batch size (0 = batching handled by vLLM)
max_batch_size: 0

# Input configuration
input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "image"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }}
]

# Output configuration
output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

# Instance configuration
instance_group [
  {{
    count: 1
    kind: KIND_MODEL
  }}
]
'''


def main():
    parser = argparse.ArgumentParser(
        description="Set up Triton model repository for Qwen2-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to merged model (default: {DEFAULT_MODEL_PATH})"
    )

    parser.add_argument(
        "--repo", "-r",
        type=str,
        default=DEFAULT_REPO_PATH,
        help=f"Triton model repository path (default: {DEFAULT_REPO_PATH})"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name in Triton (default: {DEFAULT_MODEL_NAME})"
    )

    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU (default: 1)"
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)"
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Model dtype (default: bfloat16)"
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    # Check for model files
    safetensors = list(model_path.glob("*.safetensors"))
    if not safetensors:
        print(f"Error: No safetensor files found in: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("Triton Model Repository Setup")
    print("=" * 60)
    print(f"Model path: {args.model}")
    print(f"Repository: {args.repo}")
    print(f"Model name: {args.model_name}")

    # Create repository structure
    repo_path = Path(args.repo)
    model_dir = repo_path / args.model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating repository structure at: {repo_path}")

    # Create model.json
    model_json = create_model_json(
        model_path=str(args.model),
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    model_json_path = model_dir / "model.json"
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=2)

    print(f"\nCreated: {model_json_path}")
    print(json.dumps(model_json, indent=2))

    # Create config.pbtxt
    config_pbtxt = create_config_pbtxt(args.model_name)
    config_path = repo_path / args.model_name / "config.pbtxt"
    with open(config_path, "w") as f:
        f.write(config_pbtxt)

    print(f"\nCreated: {config_path}")

    # Print repository structure
    print("\n" + "=" * 60)
    print("Repository Structure:")
    print("=" * 60)
    print(f"{repo_path}/")
    for p in sorted(repo_path.rglob("*")):
        rel = p.relative_to(repo_path)
        indent = "  " * len(rel.parts)
        if p.is_file():
            size = p.stat().st_size
            print(f"{indent}{p.name} ({size} bytes)")
        else:
            print(f"{indent}{p.name}/")

    # Print launch command
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo launch Triton server, run:")
    print()

    docker_cmd = f"""docker run --gpus all --rm -it \\
    --shm-size=4G \\
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
    -v {args.model}:{args.model}:ro \\
    -v {repo_path}:/models \\
    nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \\
    tritonserver --model-repository=/models"""

    print(docker_cmd)

    print("\n" + "=" * 60)
    print("API Endpoints (after server starts):")
    print("=" * 60)
    print("Health:    http://localhost:8000/v2/health/live")
    print(f"Model:     http://localhost:8000/v2/models/{args.model_name}")
    print("Metrics:   http://localhost:8002/metrics")
    print("Inference: http://localhost:8000/v2/models/{}/infer".format(args.model_name))


if __name__ == "__main__":
    main()
