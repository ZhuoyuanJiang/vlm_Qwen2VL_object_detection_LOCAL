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
"""

import argparse
import asyncio
import base64
import json
import time
from functools import lru_cache
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import io

import requests
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

@lru_cache(maxsize=1)
def _load_test_dataset():
    """Load evaluation dataset split once (prefer 'val', fall back to 'validation')."""
    from datasets import load_dataset

    try:
        return load_dataset(DATASET_ID, split="val")
    except ValueError:
        return load_dataset(DATASET_ID, split="validation")


def load_test_image(image_index: int = 0) -> str:
    """Load a test image and return as base64 string."""
    # Use the same dataset and split as evaluate_vllm_accuracy.py
    dataset = _load_test_dataset()
    image = dataset[image_index % len(dataset)]["image"]

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_http_payload(text_input: str, image_b64: str, temperature: float = 0.0, max_tokens: int = 100) -> dict:
    """
    Create HTTP inference request payload for Triton.

    IMPORTANT: The input names, shapes, and datatypes here MUST match the
    config.pbtxt in your Triton model repository. If they don't match,
    Triton will reject the request with a validation error.

    Expected config.pbtxt inputs:
        input { name: "text_input"           data_type: TYPE_STRING dims: [1] }
        input { name: "image"                data_type: TYPE_STRING dims: [1] }
        input { name: "sampling_parameters"  data_type: TYPE_STRING dims: [1] optional: true }
        input { name: "stream"               data_type: TYPE_BOOL   dims: [1] optional: true }

    Mapping: config.pbtxt TYPE_STRING -> HTTP "BYTES" datatype
    """
    return {
        "inputs": [
            {
                "name": "text_input",       # Must match config.pbtxt input name
                "shape": [1],               # Must match config.pbtxt dims
                "datatype": "BYTES",        # TYPE_STRING in pbtxt = BYTES in HTTP
                "data": [text_input]
            },
            {
                "name": "image",            # Must match config.pbtxt input name
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_b64]         # Base64-encoded image
            },
            {
                "name": "sampling_parameters",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps({"temperature": temperature, "max_tokens": max_tokens})]
            },
            {
                "name": "stream",           # Controls output streaming
                "shape": [1],
                "datatype": "BOOL",         # TYPE_BOOL in pbtxt = BOOL in HTTP
                "data": [False]             # False = return complete response
            }
        ]
    }


async def benchmark_http_request(
    session: requests.Session,
    url: str,
    payload: dict,
    request_id: int
) -> BenchmarkResult:
    """Send a single HTTP inference request."""
    start_time = time.perf_counter()
    try:
        response = session.post(url, json=payload, timeout=60)
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            # Extract text output from Triton response
            outputs = result.get("outputs", [])
            text_output = None
            for output in outputs:
                if output.get("name") == "text_output":
                    text_output = output.get("data", [None])[0]
                    break

            return BenchmarkResult(
                request_id=request_id,
                success=True,
                latency_ms=latency_ms,
                output=text_output,
                error=None
            )
        else:
            return BenchmarkResult(
                request_id=request_id,
                success=False,
                latency_ms=latency_ms,
                output=None,
                error=f"HTTP {response.status_code}: {response.text[:200]}"
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


async def run_http_benchmark(config: BenchmarkConfig, images: list[str], prompt: str) -> list[BenchmarkResult]:
    """Run HTTP benchmark with specified concurrency."""
    url = f"{config.http_url}/v2/models/{config.model_name}/infer"
    session = requests.Session()

    results = []
    semaphore = asyncio.Semaphore(config.concurrency)

    async def bounded_request(request_id: int):
        async with semaphore:
            image_b64 = images[request_id % len(images)] if config.vary_images else images[0]
            payload = create_http_payload(prompt, image_b64)
            return await benchmark_http_request(session, url, payload, request_id)

    # Create all tasks
    tasks = [bounded_request(i) for i in range(config.num_requests)]

    # Run with progress indication
    print(f"Running {config.num_requests} HTTP requests with concurrency={config.concurrency}...")
    results = await asyncio.gather(*tasks)

    session.close()
    return results


def run_grpc_benchmark(config: BenchmarkConfig, images: list[str], prompt: str) -> list[BenchmarkResult]:
    """Run gRPC benchmark."""
    if not GRPC_AVAILABLE:
        print("gRPC not available. Install tritonclient[all].")
        return []

    results = []

    try:
        client = grpcclient.InferenceServerClient(url=config.grpc_url)

        for i in range(config.num_requests):
            image_b64 = images[i % len(images)] if config.vary_images else images[0]

            # Prepare inputs
            inputs = [
                grpcclient.InferInput("text_input", [1], "BYTES"),
                grpcclient.InferInput("image", [1], "BYTES"),
                grpcclient.InferInput("sampling_parameters", [1], "BYTES"),
                grpcclient.InferInput("stream", [1], "BOOL"),
            ]

            inputs[0].set_data_from_numpy(np.array([prompt.encode()], dtype=np.object_))
            inputs[1].set_data_from_numpy(np.array([image_b64.encode()], dtype=np.object_))
            inputs[2].set_data_from_numpy(np.array([b'{"temperature": 0, "max_tokens": 100}'], dtype=np.object_))
            inputs[3].set_data_from_numpy(np.array([False], dtype=np.bool_))

            outputs = [grpcclient.InferRequestedOutput("text_output")]

            start_time = time.perf_counter()
            try:
                response = client.infer(config.model_name, inputs, outputs=outputs)
                latency_ms = (time.perf_counter() - start_time) * 1000

                text_output = response.as_numpy("text_output")[0].decode()
                results.append(BenchmarkResult(
                    request_id=i,
                    success=True,
                    latency_ms=latency_ms,
                    output=text_output,
                    error=None
                ))
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                results.append(BenchmarkResult(
                    request_id=i,
                    success=False,
                    latency_ms=latency_ms,
                    output=None,
                    error=str(e)
                ))

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{config.num_requests} requests...")

        client.close()
    except Exception as e:
        print(f"gRPC benchmark failed: {e}")

    return results


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
    images = [load_test_image(i) for i in range(min(num_images, 20))]
    print(f"Loaded {len(images)} test image(s)")

    # Prompt for nutrition detection
    prompt = "Detect the nutrition facts table in this image and return the bounding box coordinates."

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
                results = run_grpc_benchmark(config, images, prompt)

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
