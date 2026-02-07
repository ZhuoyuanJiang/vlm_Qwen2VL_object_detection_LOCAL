#!/usr/bin/env python3
"""
Benchmark script for Triton Inference Server with vLLM backend.

This script benchmarks the Qwen2-VL nutrition detection model deployed on Triton,
supporting both HTTP and gRPC endpoints.

Usage:
    # HTTP benchmark
    python benchmark_triton.py --endpoint http --model qwen2vl_nutrition_gptq_int4

    # gRPC benchmark
    python benchmark_triton.py --endpoint grpc --model qwen2vl_nutrition_gptq_int4

    # Compare both models
    python benchmark_triton.py --model qwen2vl_nutrition_bf16 qwen2vl_nutrition_gptq_int4

    # Higher concurrency (true async HTTP)
    python benchmark_triton.py --endpoint http --concurrency 20

    # Higher concurrency (true async gRPC)
    python benchmark_triton.py --endpoint grpc --concurrency 20

    # If processor files aren't cached locally, point to a local model path
    QWEN2VL_PROCESSOR_PATH=/path/to/merged/model python benchmark_triton.py --endpoint http
"""

import argparse
import asyncio
import base64
import json
import os
import time
from functools import lru_cache
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import io

import aiohttp
import numpy as np
from PIL import Image

# Optional: gRPC support
try:
    import tritonclient.grpc as grpcclient
    import tritonclient.grpc.aio as grpcclient_aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("Warning: tritonclient not installed. gRPC benchmarks disabled.")
    print("Install with: pip install tritonclient[all]")


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_name: str
    endpoint: str  # "http" or "grpc"
    http_url: str
    grpc_url: str
    num_requests: int
    concurrency: int
    vary_images: bool
    output_path: Optional[str]


@dataclass
class BenchmarkResult:
    """Single request result."""
    request_id: int
    success: bool
    latency_ms: float
    output: Optional[str]
    error: Optional[str]


@dataclass
class BenchmarkSummary:
    """Overall benchmark summary."""
    model_name: str
    endpoint: str
    num_requests: int
    concurrency: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    throughput_rps: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float


DATASET_ID = "openfoodfacts/nutrition-table-detection"

# Training prompts (from src/data/dataset.py)
TRAINING_SYSTEM_PROMPT = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
TRAINING_USER_PROMPT = "Detect the bounding box of the nutrition table."

PROCESSOR_ENV_VAR = "QWEN2VL_PROCESSOR_PATH"
DEFAULT_PROCESSOR_ID = "Qwen/Qwen2-VL-7B-Instruct"


@lru_cache(maxsize=1)
def _load_processor():
    try:
        from transformers import Qwen2VLProcessor
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: transformers. Install it to use chat templates."
        ) from e

    model_id = os.getenv(PROCESSOR_ENV_VAR, DEFAULT_PROCESSOR_ID)
    try:
        return Qwen2VLProcessor.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Qwen2VLProcessor from '{model_id}'. "
            f"Set {PROCESSOR_ENV_VAR} to a local model path."
        ) from e


@lru_cache(maxsize=1)
def _load_test_dataset():
    """Load evaluation dataset split once (prefer 'val', fall back to 'validation')."""
    from datasets import load_dataset

    try:
        return load_dataset(DATASET_ID, split="val")
    except ValueError:
        return load_dataset(DATASET_ID, split="validation")


def load_test_image(image_index: int = 0) -> tuple[Image.Image, str]:
    """Load a test image and return (PIL image, base64 string)."""
    # Use the same dataset and split as evaluate_vllm_accuracy.py
    dataset = _load_test_dataset()
    image = dataset[image_index % len(dataset)]["image"]

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image, image_b64


def build_chat_template_prompt(
    image: Image.Image,
    system_prompt: str = TRAINING_SYSTEM_PROMPT,
    user_prompt: str = TRAINING_USER_PROMPT,
) -> str:
    """Build the text_input using Qwen2-VL's chat template (matches training)."""
    processor = _load_processor()
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def create_http_payload(text_input: str, image_b64: str, temperature: float = 0.0, max_tokens: int = 100) -> dict:
    """
    Create HTTP request payload for Triton's /generate endpoint.

    The /generate endpoint uses a flat JSON format (not the structured /infer format).
    This is required because the vLLM backend uses decoupled transaction policy
    (model_transaction_policy { decoupled: true }) for streaming support, and the
    /infer endpoint doesn't support decoupled models.

    The /generate endpoint accepts the same input names as config.pbtxt but in a
    flat key-value format, plus sampling parameters as top-level keys.
    """
    return {
        "text_input": text_input,
        "image": image_b64,
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
    }


async def benchmark_http_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int
) -> BenchmarkResult:
    """Send a single HTTP inference request (async)."""
    start_time = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status == 200:
                result = await response.json()
                # /generate endpoint returns {"text_output": "...", ...}
                text_output = result.get("text_output")

                return BenchmarkResult(
                    request_id=request_id,
                    success=True,
                    latency_ms=latency_ms,
                    output=text_output,
                    error=None
                )
            else:
                body = await response.text()
                return BenchmarkResult(
                    request_id=request_id,
                    success=False,
                    latency_ms=latency_ms,
                    output=None,
                    error=f"HTTP {response.status}: {body[:200]}"
                )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return BenchmarkResult(
            request_id=request_id,
            success=False,
            latency_ms=latency_ms,
            output=None,
            error=str(e)
        )


async def run_http_benchmark(
    config: BenchmarkConfig,
    images: list[tuple[Image.Image, str]],
    prompt: str,
) -> list[BenchmarkResult]:
    """Run HTTP benchmark with specified concurrency (true async)."""
    url = f"{config.http_url}/v2/models/{config.model_name}/generate"
    connector = aiohttp.TCPConnector(limit=config.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    semaphore = asyncio.Semaphore(config.concurrency)

    async def bounded_request(request_id: int):
        async with semaphore:
            image_b64 = images[request_id % len(images)][1] if config.vary_images else images[0][1]
            payload = create_http_payload(prompt, image_b64)
            return await benchmark_http_request(session, url, payload, request_id)

    # Create all tasks
    tasks = [bounded_request(i) for i in range(config.num_requests)]

    # Run with progress indication
    print(f"Running {config.num_requests} HTTP requests with concurrency={config.concurrency}...")
    results = await asyncio.gather(*tasks)

    await session.close()
    return results


async def run_grpc_benchmark(
    config: BenchmarkConfig,
    images: list[tuple[Image.Image, str]],
    prompt: str,
) -> list[BenchmarkResult]:
    """Run gRPC benchmark with async client + concurrency."""
    if not GRPC_AVAILABLE:
        print("gRPC not available. Install tritonclient[all].")
        return []

    try:
        client = grpcclient_aio.InferenceServerClient(url=config.grpc_url)
        semaphore = asyncio.Semaphore(config.concurrency)

        async def bounded_request(request_id: int) -> BenchmarkResult:
            async with semaphore:
                image_b64 = images[request_id % len(images)][1] if config.vary_images else images[0][1]

                # Prepare inputs
                inputs = [
                    grpcclient_aio.InferInput("text_input", [1], "BYTES"),
                    grpcclient_aio.InferInput("image", [1], "BYTES"),
                    grpcclient_aio.InferInput("sampling_parameters", [1], "BYTES"),
                    grpcclient_aio.InferInput("stream", [1], "BOOL"),
                ]

                inputs[0].set_data_from_numpy(np.array([prompt.encode()], dtype=np.object_))
                inputs[1].set_data_from_numpy(np.array([image_b64.encode()], dtype=np.object_))
                inputs[2].set_data_from_numpy(np.array([b'{"temperature": 0, "max_tokens": 100}'], dtype=np.object_))
                inputs[3].set_data_from_numpy(np.array([False], dtype=np.bool_))

                outputs = [grpcclient_aio.InferRequestedOutput("text_output")]

                start_time = time.perf_counter()
                try:
                    response = await client.infer(config.model_name, inputs, outputs=outputs)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    text_output = response.as_numpy("text_output")[0].decode()
                    return BenchmarkResult(
                        request_id=request_id,
                        success=True,
                        latency_ms=latency_ms,
                        output=text_output,
                        error=None
                    )
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return BenchmarkResult(
                        request_id=request_id,
                        success=False,
                        latency_ms=latency_ms,
                        output=None,
                        error=str(e)
                    )

        tasks = [bounded_request(i) for i in range(config.num_requests)]
        results = await asyncio.gather(*tasks)

        close_result = client.close()
        if asyncio.iscoroutine(close_result):
            await close_result

        return results
    except Exception as e:
        print(f"gRPC benchmark failed: {e}")
        return []


def calculate_summary(config: BenchmarkConfig, results: list[BenchmarkResult], total_time: float) -> BenchmarkSummary:
    """Calculate benchmark summary statistics."""
    successful = [r for r in results if r.success]
    latencies = sorted([r.latency_ms for r in successful])

    if not latencies:
        latencies = [0]

    def percentile(data: list, p: float) -> float:
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]

    return BenchmarkSummary(
        model_name=config.model_name,
        endpoint=config.endpoint,
        num_requests=config.num_requests,
        concurrency=config.concurrency,
        successful_requests=len(successful),
        failed_requests=len(results) - len(successful),
        total_time_s=total_time,
        throughput_rps=len(successful) / total_time if total_time > 0 else 0,
        avg_latency_ms=sum(latencies) / len(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        p50_latency_ms=percentile(latencies, 50),
        p90_latency_ms=percentile(latencies, 90),
        p99_latency_ms=percentile(latencies, 99),
    )


def print_summary(summary: BenchmarkSummary):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {summary.model_name} ({summary.endpoint.upper()})")
    print("=" * 60)
    print(f"\n[Configuration]")
    print(f"  Requests:    {summary.num_requests}")
    print(f"  Concurrency: {summary.concurrency}")
    print(f"\n[Results]")
    print(f"  Successful:  {summary.successful_requests}/{summary.num_requests}")
    print(f"  Failed:      {summary.failed_requests}")
    print(f"  Total time:  {summary.total_time_s:.2f}s")
    print(f"  Throughput:  {summary.throughput_rps:.2f} req/s")
    print(f"\n[Latency (successful requests)]")
    print(f"  Avg:  {summary.avg_latency_ms:.1f} ms")
    print(f"  Min:  {summary.min_latency_ms:.1f} ms")
    print(f"  Max:  {summary.max_latency_ms:.1f} ms")
    print(f"  P50:  {summary.p50_latency_ms:.1f} ms")
    print(f"  P90:  {summary.p90_latency_ms:.1f} ms")
    print(f"  P99:  {summary.p99_latency_ms:.1f} ms")
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton Inference Server")
    parser.add_argument("--model", type=str, nargs="+",
                        default=["qwen2vl_nutrition_gptq_int4"],
                        help="Model name(s) to benchmark")
    parser.add_argument("--endpoint", type=str, choices=["http", "grpc", "both"],
                        default="http", help="Endpoint type")
    parser.add_argument("--http-url", type=str, default="http://localhost:8000",
                        help="Triton HTTP URL")
    parser.add_argument("--grpc-url", type=str, default="localhost:8001",
                        help="Triton gRPC URL")
    parser.add_argument("--num-requests", type=int, default=20,
                        help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests")
    parser.add_argument("--vary-images", action="store_true",
                        help="Use different images for each request")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    # Load test images
    print("Loading test images...")
    num_images = args.num_requests if args.vary_images else 1
    images = [load_test_image(i) for i in range(num_images)]
    print(f"Loaded {len(images)} test image(s)")

    # Prompt for nutrition detection (aligned with training via chat template)
    prompt = build_chat_template_prompt(images[0][0])

    all_summaries = []

    for model_name in args.model:
        endpoints = ["http", "grpc"] if args.endpoint == "both" else [args.endpoint]

        for endpoint in endpoints:
            if endpoint == "grpc" and not GRPC_AVAILABLE:
                print(f"Skipping gRPC benchmark for {model_name} (tritonclient not installed)")
                continue

            config = BenchmarkConfig(
                model_name=model_name,
                endpoint=endpoint,
                http_url=args.http_url,
                grpc_url=args.grpc_url,
                num_requests=args.num_requests,
                concurrency=args.concurrency,
                vary_images=args.vary_images,
                output_path=args.output,
            )

            print(f"\n{'=' * 60}")
            print(f"Benchmarking {model_name} via {endpoint.upper()}")
            print(f"{'=' * 60}")

            start_time = time.perf_counter()

            if endpoint == "http":
                results = await run_http_benchmark(config, images, prompt)
            else:
                results = await run_grpc_benchmark(config, images, prompt)

            total_time = time.perf_counter() - start_time

            summary = calculate_summary(config, results, total_time)
            print_summary(summary)
            all_summaries.append(summary)

    # Save results
    if args.output:
        output_data = {
            "config": {
                "models": args.model,
                "endpoint": args.endpoint,
                "num_requests": args.num_requests,
                "concurrency": args.concurrency,
                "vary_images": args.vary_images,
            },
            "summaries": [asdict(s) for s in all_summaries]
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
