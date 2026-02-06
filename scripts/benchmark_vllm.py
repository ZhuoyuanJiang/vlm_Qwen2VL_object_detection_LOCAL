"""
vLLM Benchmark Script for Performance Metrics Collection

This script benchmarks vLLM serving performance by:
1. Sending requests at different concurrency levels
2. Collecting TTFT, TPOT, E2E metrics from /metrics endpoint
3. Measuring client-side latencies for comparison

Metrics collected:
- TTFT (Time To First Token): Dominated by prefill phase
- TPOT (Time Per Output Token): Reflects decode speed
- E2E (End-to-End Latency): Total request time

Usage:
    python scripts/benchmark_vllm.py --num-requests 10 --concurrency 1
    python scripts/benchmark_vllm.py --num-requests 20 --concurrency 4

Note:
    This script's prompts are aligned to training (system + user). If you
    benchmarked with older prompts, rerun for apples-to-apples comparison.
"""

import argparse
import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
import requests
from datasets import load_dataset

# VRAM monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not installed. VRAM monitoring disabled.")

# Handle nested event loops (for Jupyter notebooks)
# Jupyter already runs an event loop, so asyncio.run() fails without this patch
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio not installed, will work in CLI but not in notebooks


# =============================================================================
# Configuration
# =============================================================================
VLLM_HOST = "localhost"
VLLM_PORT = 8000  # Default, can be overridden via --port
MODEL_NAME = "qwen2vl-nutrition"

SYSTEM_PROMPT = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

USER_PROMPT = "Detect the bounding box of the nutrition table."


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class RequestResult:
    """Result from a single inference request."""
    request_id: int
    success: bool
    client_latency_ms: float  # Client-side measured latency
    output_text: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MetricsSnapshot:
    """Snapshot of vLLM metrics from /metrics endpoint."""
    ttft_sum: float
    ttft_count: int
    tpot_sum: float
    tpot_count: int
    e2e_sum: float
    e2e_count: int
    prefill_time_sum: float
    prefill_count: int
    decode_time_sum: float
    decode_count: int
    itl_sum: float  # Inter-token latency
    itl_count: int
    prompt_tokens: int
    generation_tokens: int
    requests_running: int
    requests_waiting: int
    kv_cache_usage: float


@dataclass
class MemoryProfile:
    """GPU memory profile at different stages."""
    post_load_mb: float      # After model loads, before any inference
    post_warmup_mb: float    # After warmup requests complete
    peak_mb: float           # Maximum during benchmark run
    kv_cache_usage_pct: float  # From vLLM /metrics


# =============================================================================
# Memory Monitoring
# =============================================================================
def get_gpu_memory_mb(device_index: int = 0) -> float:
    """Get current GPU memory usage in MB using pynvml."""
    if not PYNVML_AVAILABLE:
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)  # Convert to MB
    except Exception as e:
        print(f"Warning: Could not get GPU memory: {e}")
        return 0.0
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def get_gpu_memory_total_mb(device_index: int = 0) -> float:
    """Get total GPU memory in MB."""
    if not PYNVML_AVAILABLE:
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total / (1024 ** 2)
    except:
        return 0.0
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


# =============================================================================
# Metrics Collection
# =============================================================================
def parse_prometheus_metric(lines: list, metric_name: str, suffix: str = "") -> float:
    """Parse a specific metric value from Prometheus format."""
    target = f"{metric_name}{suffix}"
    for line in lines:
        if line.startswith(target) and "{" in line:
            # Handle metrics with labels like vllm:metric{engine="0",...} value
            parts = line.split("} ")
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
        elif line.startswith(target) and not line.startswith("#"):
            # Simple metric without labels
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return 0.0


def get_metrics_snapshot() -> MetricsSnapshot:
    """Fetch current metrics from vLLM /metrics endpoint."""
    response = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/metrics")
    lines = response.text.split("\n")

    return MetricsSnapshot(
        ttft_sum=parse_prometheus_metric(lines, "vllm:time_to_first_token_seconds_sum"),
        ttft_count=int(parse_prometheus_metric(lines, "vllm:time_to_first_token_seconds_count")),
        tpot_sum=parse_prometheus_metric(lines, "vllm:request_time_per_output_token_seconds_sum"),
        tpot_count=int(parse_prometheus_metric(lines, "vllm:request_time_per_output_token_seconds_count")),
        e2e_sum=parse_prometheus_metric(lines, "vllm:e2e_request_latency_seconds_sum"),
        e2e_count=int(parse_prometheus_metric(lines, "vllm:e2e_request_latency_seconds_count")),
        prefill_time_sum=parse_prometheus_metric(lines, "vllm:request_prefill_time_seconds_sum"),
        prefill_count=int(parse_prometheus_metric(lines, "vllm:request_prefill_time_seconds_count")),
        decode_time_sum=parse_prometheus_metric(lines, "vllm:request_decode_time_seconds_sum"),
        decode_count=int(parse_prometheus_metric(lines, "vllm:request_decode_time_seconds_count")),
        itl_sum=parse_prometheus_metric(lines, "vllm:inter_token_latency_seconds_sum"),
        itl_count=int(parse_prometheus_metric(lines, "vllm:inter_token_latency_seconds_count")),
        prompt_tokens=int(parse_prometheus_metric(lines, "vllm:prompt_tokens_total")),
        generation_tokens=int(parse_prometheus_metric(lines, "vllm:generation_tokens_total")),
        requests_running=int(parse_prometheus_metric(lines, "vllm:num_requests_running")),
        requests_waiting=int(parse_prometheus_metric(lines, "vllm:num_requests_waiting")),
        kv_cache_usage=parse_prometheus_metric(lines, "vllm:kv_cache_usage_perc"),
    )


def compute_metrics_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> dict:
    """Compute the difference in metrics between two snapshots."""
    num_requests = after.ttft_count - before.ttft_count

    if num_requests == 0:
        return {"error": "No new requests recorded"}

    ttft_delta = after.ttft_sum - before.ttft_sum
    tpot_delta = after.tpot_sum - before.tpot_sum
    e2e_delta = after.e2e_sum - before.e2e_sum
    prefill_delta = after.prefill_time_sum - before.prefill_time_sum
    decode_delta = after.decode_time_sum - before.decode_time_sum
    itl_delta = after.itl_sum - before.itl_sum
    itl_count_delta = after.itl_count - before.itl_count

    return {
        "num_requests": num_requests,
        "avg_ttft_ms": (ttft_delta / num_requests) * 1000,
        "avg_tpot_ms": (tpot_delta / num_requests) * 1000,
        "avg_e2e_ms": (e2e_delta / num_requests) * 1000,
        "avg_prefill_ms": (prefill_delta / num_requests) * 1000,
        "avg_decode_ms": (decode_delta / num_requests) * 1000,
        "avg_itl_ms": (itl_delta / itl_count_delta * 1000) if itl_count_delta > 0 else 0,  # Inter-token latency
        "total_prompt_tokens": after.prompt_tokens - before.prompt_tokens,
        "total_generation_tokens": after.generation_tokens - before.generation_tokens,
    }


# =============================================================================
# Async Request Functions
# =============================================================================
async def send_request_async(
    session: aiohttp.ClientSession,
    request_id: int,
    image_b64: str,
) -> RequestResult:
    """Send a single async request to vLLM."""
    start_time = time.perf_counter()

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
        "temperature": 0.0,
        "skip_special_tokens": False
    }

    try:
        async with session.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            result = await response.json()
            end_time = time.perf_counter()

            if "choices" in result:
                return RequestResult(
                    request_id=request_id,
                    success=True,
                    client_latency_ms=(end_time - start_time) * 1000,
                    output_text=result["choices"][0]["message"]["content"]
                )
            else:
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    client_latency_ms=(end_time - start_time) * 1000,
                    error=str(result)
                )
    except Exception as e:
        end_time = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            success=False,
            client_latency_ms=(end_time - start_time) * 1000,
            error=str(e)
        )


async def run_benchmark_async(
    num_requests: int,
    concurrency: int,
    image_b64_list: list[str],
) -> list[RequestResult]:
    """Run benchmark with specified concurrency level."""
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(request_id: int) -> RequestResult:
        async with semaphore:
            # Each request gets its corresponding image from the list
            return await send_request_async(session, request_id, image_b64_list[request_id])

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

    return results


# =============================================================================
# Main Benchmark Function
# =============================================================================
def run_benchmark(
    num_requests: int = 10,
    concurrency: int = 1,
    vary_images: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run vLLM benchmark and collect metrics.

    Args:
        num_requests: Total number of requests to send
        concurrency: Number of concurrent requests
        vary_images: If True, use different images for each request (realistic scenario)
                    If False, use same image for all requests (best-case prefix caching)
        verbose: Print progress and results

    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        print("=" * 60)
        print(f"vLLM Benchmark: {num_requests} requests @ concurrency={concurrency}")
        print(f"  Image mode: {'DIFFERENT images (realistic)' if vary_images else 'SAME image (best-case caching)'}")
        print("=" * 60)

    # Load test images
    if verbose:
        print("\n1. Loading test images from HuggingFace dataset...")
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")

    if vary_images:
        # Load different images for each request
        num_unique_images = min(num_requests, len(ds))
        images_b64 = []
        for i in range(num_unique_images):
            img = ds[i]['image']
            img.save(f'/tmp/benchmark_test_{i}.jpg')
            with open(f'/tmp/benchmark_test_{i}.jpg', 'rb') as f:
                images_b64.append(base64.b64encode(f.read()).decode())
        if verbose:
            print(f"   Loaded {num_unique_images} different images")
        # Cycle through images if num_requests > num_unique_images
        image_b64_list = [images_b64[i % num_unique_images] for i in range(num_requests)]
    else:
        # Use same image for all requests
        test_image = ds[0]['image']
        test_image.save('/tmp/benchmark_test.jpg')
        with open('/tmp/benchmark_test.jpg', 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        image_b64_list = [image_b64] * num_requests
        if verbose:
            print(f"   Using same image for all requests (size: {test_image.size})")

    # Capture initial memory (post-load, before any requests from this benchmark)
    if verbose:
        print("\n2. Capturing initial VRAM and metrics...")
    initial_memory_mb = get_gpu_memory_mb()
    metrics_before = get_metrics_snapshot()

    if verbose:
        print(f"   Initial VRAM usage: {initial_memory_mb:.0f} MB ({initial_memory_mb/1024:.2f} GB)")

    # Track peak memory during benchmark
    peak_memory_mb = initial_memory_mb

    # Run benchmark
    if verbose:
        print(f"\n3. Running {num_requests} requests with concurrency={concurrency}...")

    start_time = time.perf_counter()
    results = asyncio.run(run_benchmark_async(num_requests, concurrency, image_b64_list))
    total_time = time.perf_counter() - start_time

    # Capture memory after benchmark (this is likely the peak)
    post_benchmark_memory_mb = get_gpu_memory_mb()
    peak_memory_mb = max(peak_memory_mb, post_benchmark_memory_mb)

    # Get metrics after benchmark
    if verbose:
        print("\n4. Capturing final metrics and VRAM...")
    metrics_after = get_metrics_snapshot()
    final_memory_mb = get_gpu_memory_mb()

    if verbose:
        print(f"   Final VRAM usage: {final_memory_mb:.0f} MB ({final_memory_mb/1024:.2f} GB)")
        print(f"   Peak VRAM usage: {peak_memory_mb:.0f} MB ({peak_memory_mb/1024:.2f} GB)")
        print(f"   KV Cache usage: {metrics_after.kv_cache_usage:.1%}")

    # Compute results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    client_latencies = [r.client_latency_ms for r in successful]
    avg_client_latency = sum(client_latencies) / len(client_latencies) if client_latencies else 0

    # Compute server-side metrics delta
    server_metrics = compute_metrics_delta(metrics_before, metrics_after)

    # Compute throughput
    throughput = num_requests / total_time if total_time > 0 else 0

    # Build memory profile
    memory_profile = {
        "initial_mb": initial_memory_mb,
        "initial_gb": initial_memory_mb / 1024,
        "peak_mb": peak_memory_mb,
        "peak_gb": peak_memory_mb / 1024,
        "final_mb": final_memory_mb,
        "final_gb": final_memory_mb / 1024,
        "kv_cache_usage_pct": metrics_after.kv_cache_usage,
    }

    # Build results dictionary
    benchmark_results = {
        "config": {
            "num_requests": num_requests,
            "concurrency": concurrency,
            "vary_images": vary_images,
        },
        "summary": {
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_time_s": total_time,
            "throughput_rps": throughput,
        },
        "client_metrics": {
            "avg_latency_ms": avg_client_latency,
            "min_latency_ms": min(client_latencies) if client_latencies else 0,
            "max_latency_ms": max(client_latencies) if client_latencies else 0,
        },
        "server_metrics": server_metrics,
        "memory_profile": memory_profile,
    }

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\n[Configuration]")
        print(f"  Requests: {num_requests}")
        print(f"  Concurrency: {concurrency}")

        print(f"\n[Summary]")
        print(f"  Successful: {len(successful)}/{num_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        print(f"\n[Client-Side Latency]")
        print(f"  Avg: {avg_client_latency:.1f} ms")
        print(f"  Min: {min(client_latencies):.1f} ms" if client_latencies else "  Min: N/A")
        print(f"  Max: {max(client_latencies):.1f} ms" if client_latencies else "  Max: N/A")

        print(f"\n[Server-Side Metrics (from /metrics)]")
        if "error" not in server_metrics:
            print(f"  Avg TTFT: {server_metrics['avg_ttft_ms']:.1f} ms")
            print(f"  Avg TPOT: {server_metrics['avg_tpot_ms']:.1f} ms")
            print(f"  Avg ITL:  {server_metrics['avg_itl_ms']:.1f} ms (inter-token latency)")
            print(f"  Avg E2E:  {server_metrics['avg_e2e_ms']:.1f} ms")
            print(f"  Avg Prefill: {server_metrics['avg_prefill_ms']:.1f} ms")
            print(f"  Avg Decode:  {server_metrics['avg_decode_ms']:.1f} ms")
            print(f"  Tokens (prompt/gen): {server_metrics['total_prompt_tokens']}/{server_metrics['total_generation_tokens']}")
        else:
            print(f"  Error: {server_metrics['error']}")

        print(f"\n[VRAM Usage]")
        print(f"  Initial: {memory_profile['initial_gb']:.2f} GB")
        print(f"  Peak:    {memory_profile['peak_gb']:.2f} GB")
        print(f"  Final:   {memory_profile['final_gb']:.2f} GB")
        print(f"  KV Cache: {memory_profile['kv_cache_usage_pct']:.1%}")

        print("\n" + "=" * 60)

    return benchmark_results


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM serving performance")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--vary-images", action="store_true",
                       help="Use different images for each request (realistic scenario). "
                            "Default: use same image (best-case prefix caching)")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")

    args = parser.parse_args()

    # Override global port if specified
    global VLLM_PORT
    VLLM_PORT = args.port

    results = run_benchmark(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        vary_images=args.vary_images,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
