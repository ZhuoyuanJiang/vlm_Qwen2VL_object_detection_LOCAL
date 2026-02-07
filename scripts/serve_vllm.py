#!/usr/bin/env python3
"""
Launch a vLLM OpenAI-compatible API server for the Qwen2-VL model.

This script is a thin wrapper that keeps serving commands readable and easy to reuse.

Examples:
    python scripts/serve_vllm.py --gpu 0

    python scripts/serve_vllm.py \
      --gpu 0 \
      --port 8000 \
      --gpu-memory-utilization 0.9 \
      --limit-mm-per-prompt '{"image":1}'

    # Pass extra raw vLLM args when needed
    python scripts/serve_vllm.py --gpu 0 -- --max-num-seqs 256 --enable-chunked-prefill
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


DEFAULT_MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
DEFAULT_MODEL_NAME = "qwen2vl-nutrition"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LIMIT_MM_PER_PROMPT = '{"image":1}'


def wait_for_server(host: str, port: int, timeout: int = 120) -> bool:
    """Wait until /health returns 200."""
    print(f"Waiting for server to be ready (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print()
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch vLLM server for Qwen2-VL")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help="Path to model directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Served model name")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="Server port")

    parser.add_argument("--gpu", default=None, help="Set CUDA_VISIBLE_DEVICES (e.g., '0' or '0,1')")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit-mm-per-prompt", default=DEFAULT_LIMIT_MM_PER_PROMPT)
    parser.add_argument("--quantization", default=None, help="Optional quantization mode (fp8/awq/gptq)")
    parser.add_argument("--no-enable-prefix-caching", action="store_true")
    parser.add_argument("--no-wait", action="store_true", help="Skip health-check waiting")
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model path does not exist: {model_path}")
    if not list(model_path.glob("*.safetensors")):
        parser.error(f"No .safetensors files found in: {model_path}")

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using CUDA_VISIBLE_DEVICES={args.gpu}")
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"Using existing CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("CUDA_VISIBLE_DEVICES not set (vLLM will use system default visibility).")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        args.model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt",
        args.limit_mm_per_prompt,
    ]

    if args.tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.no_enable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if passthrough:
        cmd.extend(passthrough)

    print("Launch command:")
    print(shlex.join(cmd))

    process = subprocess.Popen(cmd)

    def _shutdown(_sig, _frame):
        print("\nStopping server...")
        process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        if not args.no_wait:
            host_for_check = "localhost" if args.host == "0.0.0.0" else args.host
            if wait_for_server(host_for_check, args.port):
                print("\nServer ready.")
                print(f"API:    http://localhost:{args.port}/v1/chat/completions")
                print(f"Health: http://localhost:{args.port}/health")
                print(f"Models: http://localhost:{args.port}/v1/models")
            else:
                print("\nWarning: health check did not pass before timeout.")
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
