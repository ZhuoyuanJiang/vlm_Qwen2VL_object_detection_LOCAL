#!/usr/bin/env python3
"""
Launch vLLM server for Qwen2-VL model serving.

This script provides a simple way to start the vLLM OpenAI-compatible API server
for the trained Qwen2-VL nutrition detection model.

Usage:
    # Basic usage (uses defaults)
    python scripts/serve_vllm.py

    # Custom model path and port
    python scripts/serve_vllm.py --model /path/to/merged/model --port 8000

    # Specify GPUs
    python scripts/serve_vllm.py --gpu 2,3

    # All options
    python scripts/serve_vllm.py \
        --model /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
        --port 8000 \
        --gpu 2,3 \
        --model-name qwen2vl-nutrition

Example API call after server starts:
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "qwen2vl-nutrition", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

import argparse
import os
import subprocess
import sys
import time
import signal
import requests
from pathlib import Path


# Default configuration
DEFAULT_MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
DEFAULT_PORT = 8000
DEFAULT_MODEL_NAME = "qwen2vl-nutrition"
DEFAULT_HOST = "0.0.0.0"


def find_available_gpus(max_gpus: int = 2) -> list:
    """Find available GPUs with low memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        available = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_idx = int(parts[0].strip())
            mem_used = int(parts[1].strip())
            mem_total = int(parts[2].strip())

            # Consider GPU available if <2GB used and has at least 40GB total
            if mem_used < 2000 and mem_total > 40000:
                available.append(gpu_idx)

        return available[:max_gpus]

    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}")
        return []


def wait_for_server(host: str, port: int, timeout: int = 120) -> bool:
    """Wait for vLLM server to be ready."""
    print(f"Waiting for server to be ready (timeout: {timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
        print(".", end="", flush=True)

    print()
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM server for Qwen2-VL model",
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
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name for API (default: {DEFAULT_MODEL_NAME})"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU IDs to use (e.g., '2,3'). Auto-detects if not specified."
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )

    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU (default: 1)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Model dtype (default: bfloat16)"
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server to be ready, exit immediately after launch"
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
    print("vLLM Server Launcher")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model name: {args.model_name}")
    print(f"Port: {args.port}")
    print(f"Dtype: {args.dtype}")

    # Set up GPU
    if args.gpu:
        gpu_str = args.gpu
    else:
        print("\nAuto-detecting available GPUs...")
        available = find_available_gpus(max_gpus=args.tensor_parallel_size)
        if available:
            gpu_str = ",".join(map(str, available))
            print(f"Found available GPUs: {gpu_str}")
        else:
            print("No available GPUs found, using default")
            gpu_str = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"Using GPUs: {gpu_str}")

    # Build vLLM command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(args.model),
        "--served-model-name", args.model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--trust-remote-code",
        "--max-model-len", str(args.max_model_len),
    ]

    if args.tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    print("\n" + "=" * 60)
    print("Launch Command:")
    print(" ".join(cmd))
    print("=" * 60 + "\n")

    # Launch server
    print("Starting vLLM server...")

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run in foreground so user can see logs and Ctrl+C to stop
        process = subprocess.Popen(cmd)

        if not args.no_wait:
            # Wait for server to be ready
            host = "localhost" if args.host == "0.0.0.0" else args.host
            if wait_for_server(host, args.port):
                print(f"\n{'=' * 60}")
                print("SERVER READY!")
                print(f"{'=' * 60}")
                print(f"API endpoint: http://localhost:{args.port}/v1/chat/completions")
                print(f"Health check: http://localhost:{args.port}/health")
                print(f"Models list:  http://localhost:{args.port}/v1/models")
                print(f"\nPress Ctrl+C to stop the server")
                print("=" * 60 + "\n")
            else:
                print("\nWarning: Server did not respond to health check")
                print("Check the logs above for errors")

        # Wait for process to finish (or Ctrl+C)
        process.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
        process.terminate()
        process.wait(timeout=10)
        print("Server stopped")


if __name__ == "__main__":
    main()
