# %% [markdown]
# # vLLM Performance Analysis
#
# This notebook covers:
# 1. Understanding vLLM performance metrics (TTFT, TPOT, E2E)
# 2. Collecting metrics from the `/metrics` endpoint
# 3. Running benchmarks at different concurrency levels
# 4. Interpreting results for mentor presentation
#
# ## Key Concepts
#
# ### LLM Inference Stages
#
# ```
# Request arrives
#      │
#      ▼
# ┌─────────────┐
# │   PREFILL   │  ← Process prompt tokens, populate KV cache
# │  (parallel) │    Dominates TTFT (Time To First Token)
# └─────────────┘
#      │
#      ▼
# ┌─────────────┐
# │   DECODE    │  ← Generate output tokens one-by-one
# │ (sequential)│    Dominates TPOT (Time Per Output Token)
# └─────────────┘
#      │
#      ▼
# Response complete
# ```
#
# ### Metrics Definitions
#
# | Metric | Full Name | What It Measures |
# |--------|-----------|------------------|
# | **TTFT** | Time To First Token | Latency until first output token (prefill time) |
# | **TPOT** | Time Per Output Token | Average time per generated token (decode speed) |
# | **E2E** | End-to-End Latency | Total request time (TTFT + decode time) |
#
# ### Concurrency vs Num-Requests
#
# | Parameter | Meaning |
# |-----------|---------|
# | `num-requests` | Total requests to send |
# | `concurrency` | How many run simultaneously |
#
# Example: `num-requests=20, concurrency=4` → 20 requests sent 4 at a time (5 batches)

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import subprocess
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Configuration
VLLM_HOST = "localhost"
VLLM_PORT = 8000

print(f"vLLM server: http://{VLLM_HOST}:{VLLM_PORT}")

# %% [markdown]
# ## 2. Check Server Health

# %%
# Health check
try:
    response = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=5)
    print(f"Server status: {'Healthy ✓' if response.status_code == 200 else 'Unhealthy ✗'}")
except Exception as e:
    print(f"Server not reachable: {e}")

# %%
# List available models
try:
    response = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models", timeout=5)
    models = response.json()
    print("Available models:")
    for model in models.get("data", []):
        print(f"  - {model['id']}")
except Exception as e:
    print(f"Could not list models: {e}")

# %% [markdown]
# ## 3. Understanding the /metrics Endpoint
#
# vLLM exposes Prometheus-style metrics at `/metrics`. Let's fetch and parse them.

# %%
# Fetch raw metrics (this is what `curl http://localhost:8000/metrics | grep vllm` does)
def fetch_raw_metrics():
    """Fetch raw metrics from vLLM server."""
    response = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/metrics")
    return response.text

# Show a sample of the metrics
raw_metrics = fetch_raw_metrics()
vllm_lines = [line for line in raw_metrics.split('\n') if line.startswith('vllm:')]
print(f"Total vLLM metrics: {len(vllm_lines)}")
print("\nFirst 20 metrics:")
for line in vllm_lines[:20]:
    print(line)

# %% [markdown]
# ### Key Metrics to Focus On
#
# From the raw output above, here are the metrics your mentor cares about:
#
# | Metric Name | What It Measures |
# |-------------|------------------|
# | `vllm:time_to_first_token_seconds` | TTFT - dominated by prefill |
# | `vllm:request_time_per_output_token_seconds` | TPOT - decode speed |
# | `vllm:e2e_request_latency_seconds` | E2E - total request time |
# | `vllm:request_prefill_time_seconds` | Time in prefill phase |
# | `vllm:request_decode_time_seconds` | Time in decode phase |
# | `vllm:num_requests_running` | Current concurrent requests |
# | `vllm:num_requests_waiting` | Queued requests |
# | `vllm:kv_cache_usage_perc` | KV cache utilization |

# %%
def parse_prometheus_metric(lines: list, metric_name: str) -> float:
    """Parse a specific metric value from Prometheus format."""
    for line in lines:
        if line.startswith(metric_name) and not line.startswith("#"):
            # Handle metrics with labels like vllm:metric{engine="0",...} value
            if "{" in line:
                parts = line.split("} ")
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
    return 0.0


def get_current_metrics() -> dict:
    """Get current metrics in a readable format."""
    response = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/metrics")
    lines = response.text.split("\n")

    # Extract key metrics
    ttft_sum = parse_prometheus_metric(lines, "vllm:time_to_first_token_seconds_sum")
    ttft_count = parse_prometheus_metric(lines, "vllm:time_to_first_token_seconds_count")

    tpot_sum = parse_prometheus_metric(lines, "vllm:request_time_per_output_token_seconds_sum")
    tpot_count = parse_prometheus_metric(lines, "vllm:request_time_per_output_token_seconds_count")

    e2e_sum = parse_prometheus_metric(lines, "vllm:e2e_request_latency_seconds_sum")
    e2e_count = parse_prometheus_metric(lines, "vllm:e2e_request_latency_seconds_count")

    prefill_sum = parse_prometheus_metric(lines, "vllm:request_prefill_time_seconds_sum")
    decode_sum = parse_prometheus_metric(lines, "vllm:request_decode_time_seconds_sum")

    return {
        "total_requests": int(ttft_count),
        "avg_ttft_ms": (ttft_sum / ttft_count * 1000) if ttft_count > 0 else 0,
        "avg_tpot_ms": (tpot_sum / tpot_count * 1000) if tpot_count > 0 else 0,
        "avg_e2e_ms": (e2e_sum / e2e_count * 1000) if e2e_count > 0 else 0,
        "avg_prefill_ms": (prefill_sum / ttft_count * 1000) if ttft_count > 0 else 0,
        "avg_decode_ms": (decode_sum / ttft_count * 1000) if ttft_count > 0 else 0,
        "requests_running": int(parse_prometheus_metric(lines, "vllm:num_requests_running")),
        "requests_waiting": int(parse_prometheus_metric(lines, "vllm:num_requests_waiting")),
        "kv_cache_usage": parse_prometheus_metric(lines, "vllm:kv_cache_usage_perc"),
        "prompt_tokens_total": int(parse_prometheus_metric(lines, "vllm:prompt_tokens_total")),
        "generation_tokens_total": int(parse_prometheus_metric(lines, "vllm:generation_tokens_total")),
    }


# Display current metrics
metrics = get_current_metrics()
print("Current vLLM Metrics (cumulative since server start):")
print("=" * 50)
print(f"Total requests processed: {metrics['total_requests']}")
print(f"")
print(f"Average TTFT:    {metrics['avg_ttft_ms']:.1f} ms")
print(f"Average TPOT:    {metrics['avg_tpot_ms']:.1f} ms")
print(f"Average E2E:     {metrics['avg_e2e_ms']:.1f} ms")
print(f"")
print(f"Average Prefill: {metrics['avg_prefill_ms']:.1f} ms")
print(f"Average Decode:  {metrics['avg_decode_ms']:.1f} ms")
print(f"")
print(f"Requests running: {metrics['requests_running']}")
print(f"Requests waiting: {metrics['requests_waiting']}")
print(f"KV cache usage:   {metrics['kv_cache_usage']:.1%}")

# %% [markdown]
# ## 4. Interpreting the Metrics
#
# Let's break down what these numbers mean:

# %%
display(Markdown(f"""
### Metrics Interpretation

Based on **{metrics['total_requests']} requests** processed so far:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **TTFT** | {metrics['avg_ttft_ms']:.1f} ms | Time until first token. Dominated by **prefill** (processing prompt + populating KV cache) |
| **TPOT** | {metrics['avg_tpot_ms']:.1f} ms | Time per output token. Reflects **decode speed** (~{1000/metrics['avg_tpot_ms']:.0f} tokens/sec if TPOT > 0) |
| **E2E** | {metrics['avg_e2e_ms']:.1f} ms | Total request time. E2E ≈ TTFT + (num_output_tokens × TPOT) |
| **Prefill** | {metrics['avg_prefill_ms']:.1f} ms | Actual prefill computation time |
| **Decode** | {metrics['avg_decode_ms']:.1f} ms | Actual decode computation time |

### Key Insight: Prefill vs Decode

- **Prefill time ({metrics['avg_prefill_ms']:.1f} ms)** is dominated by processing the **image** (VLM-specific)
- **Decode time ({metrics['avg_decode_ms']:.1f} ms)** is for generating ~{metrics['generation_tokens_total'] // max(metrics['total_requests'], 1)} tokens per request
- For our bbox detection task, decode is fast because output is short (~22 tokens)

### Token Statistics

- Total prompt tokens: {metrics['prompt_tokens_total']:,} ({metrics['prompt_tokens_total'] // max(metrics['total_requests'], 1):,} per request)
- Total generation tokens: {metrics['generation_tokens_total']:,} ({metrics['generation_tokens_total'] // max(metrics['total_requests'], 1):,} per request)
"""))

# %% [markdown]
# ## 5. Running Benchmarks
#
# Now let's run the benchmark script at different concurrency levels.

# %%
import sys
sys.path.insert(0, '/home/zhuoyuan/projects/vlm_Qwen2VL_object_detection')

# Import the benchmark function
from scripts.benchmark_vllm import run_benchmark, get_metrics_snapshot

# %% [markdown]
# ### Experiment 1: Baseline (Single Request)
#
# First, let's establish a baseline with sequential requests (concurrency=1).

# %%
# Run baseline benchmark
print("Running Experiment 1: Baseline (concurrency=1)")
print("-" * 50)
baseline_results = run_benchmark(num_requests=10, concurrency=1, verbose=True)

# %%
# Store results for comparison
all_results = []
all_results.append({
    "concurrency": 1,
    "num_requests": 10,
    **baseline_results["summary"],
    **baseline_results["client_metrics"],
    **baseline_results["server_metrics"]
})

# %% [markdown]
# ### Experiment 2: Concurrency Scaling
#
# Now let's test with increasing concurrency to see how latency changes under load.
# We'll test concurrency levels: 1, 2, 4, 8

# %%
# Test different concurrency levels
concurrency_levels = [1, 2, 4, 8]

for conc in concurrency_levels[1:]:  # Skip 1, we already did it
    print(f"\n{'='*60}")
    print(f"Running benchmark with concurrency={conc}")
    print("=" * 60)

    results = run_benchmark(num_requests=20, concurrency=conc, verbose=True)

    all_results.append({
        "concurrency": conc,
        "num_requests": 20,
        **results["summary"],
        **results["client_metrics"],
        **results["server_metrics"]
    })

# %% [markdown]
# ## 6. Results Analysis

# %%
# Create results DataFrame
df = pd.DataFrame(all_results)

# Display results table
print("\nBenchmark Results Summary")
print("=" * 80)

# Check if avg_itl_ms exists (may not in older runs)
cols_to_show = ["concurrency", "num_requests", "successful_requests", "throughput_rps",
                "avg_ttft_ms", "avg_tpot_ms", "avg_e2e_ms", "avg_latency_ms"]
if "avg_itl_ms" in df.columns:
    cols_to_show.insert(6, "avg_itl_ms")  # Add ITL after TPOT

display(df[cols_to_show].round(2))

# %%
# Visualization: Latency vs Concurrency
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: TTFT vs Concurrency
axes[0, 0].bar(df["concurrency"].astype(str), df["avg_ttft_ms"], color='steelblue')
axes[0, 0].set_xlabel("Concurrency")
axes[0, 0].set_ylabel("TTFT (ms)")
axes[0, 0].set_title("Time To First Token vs Concurrency")
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: TPOT vs Concurrency
axes[0, 1].bar(df["concurrency"].astype(str), df["avg_tpot_ms"], color='coral')
axes[0, 1].set_xlabel("Concurrency")
axes[0, 1].set_ylabel("TPOT (ms)")
axes[0, 1].set_title("Time Per Output Token vs Concurrency")
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Inter-Token Latency vs Concurrency
if "avg_itl_ms" in df.columns:
    axes[1, 0].bar(df["concurrency"].astype(str), df["avg_itl_ms"], color='purple')
    axes[1, 0].set_xlabel("Concurrency")
    axes[1, 0].set_ylabel("ITL (ms)")
    axes[1, 0].set_title("Inter-Token Latency vs Concurrency")
    axes[1, 0].grid(axis='y', alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'ITL data not available', ha='center', va='center')
    axes[1, 0].set_title("Inter-Token Latency vs Concurrency")

# Plot 4: Throughput vs Concurrency
axes[1, 1].bar(df["concurrency"].astype(str), df["throughput_rps"], color='seagreen')
axes[1, 1].set_xlabel("Concurrency")
axes[1, 1].set_ylabel("Throughput (req/s)")
axes[1, 1].set_title("Throughput vs Concurrency")
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/vllm_benchmark_results.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to: /tmp/vllm_benchmark_results.png")

# %%
# Visualization: Prefill vs Decode time breakdown
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], df["avg_prefill_ms"], width, label='Prefill', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], df["avg_decode_ms"], width, label='Decode', color='coral')

ax.set_xlabel("Concurrency")
ax.set_ylabel("Time (ms)")
ax.set_title("Prefill vs Decode Time by Concurrency")
ax.set_xticks(x)
ax.set_xticklabels(df["concurrency"].astype(str))
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/vllm_prefill_decode_breakdown.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to: /tmp/vllm_prefill_decode_breakdown.png")

# %% [markdown]
# ## 7. Key Findings Summary

# %%
# Generate summary for mentor
baseline = df[df["concurrency"] == 1].iloc[0] if len(df[df["concurrency"] == 1]) > 0 else df.iloc[0]
max_conc = df[df["concurrency"] == df["concurrency"].max()].iloc[0]

display(Markdown(f"""
## Summary for Mentor

### Baseline Performance (concurrency=1)
- **TTFT**: {baseline['avg_ttft_ms']:.1f} ms (time to first token)
- **TPOT**: {baseline['avg_tpot_ms']:.1f} ms (time per output token)
- **E2E**: {baseline['avg_e2e_ms']:.1f} ms (end-to-end latency)
- **Throughput**: {baseline['throughput_rps']:.2f} req/s

### At Higher Concurrency (concurrency={int(max_conc['concurrency'])})
- **TTFT**: {max_conc['avg_ttft_ms']:.1f} ms ({(max_conc['avg_ttft_ms']/baseline['avg_ttft_ms']-1)*100:+.1f}% vs baseline)
- **TPOT**: {max_conc['avg_tpot_ms']:.1f} ms ({(max_conc['avg_tpot_ms']/baseline['avg_tpot_ms']-1)*100:+.1f}% vs baseline)
- **Throughput**: {max_conc['throughput_rps']:.2f} req/s ({(max_conc['throughput_rps']/baseline['throughput_rps']-1)*100:+.1f}% vs baseline)

### Key Observations

1. **Prefill dominates TTFT**: For VLMs with image input, prefill time is significant due to image processing

2. **Decode is fast**: Output tokens are generated quickly (~{1000/baseline['avg_tpot_ms']:.0f} tokens/sec) because:
   - Our bbox output is short (~22 tokens)
   - KV cache reuse makes decode efficient

3. **Concurrency tradeoff**:
   - Higher concurrency → Higher throughput
   - But also → Higher per-request latency
   - vLLM batches tokens across concurrent requests

### How to Report to Mentor

> "At concurrency={int(baseline['concurrency'])}, our model achieves {baseline['avg_ttft_ms']:.0f}ms TTFT and {baseline['avg_e2e_ms']:.0f}ms E2E latency.
> At concurrency={int(max_conc['concurrency'])}, throughput increases to {max_conc['throughput_rps']:.2f} req/s
> ({max_conc['throughput_rps']/baseline['throughput_rps']:.1f}x baseline)."
"""))

# %% [markdown]
# ## 8. Save Results

# %%
# Save results to CSV
results_file = "/home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/refactor_documentation/vllm_benchmark_results.csv"
df.to_csv(results_file, index=False)
print(f"Results saved to: {results_file}")

# Save as JSON too
json_file = "/home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/refactor_documentation/vllm_benchmark_results.json"
df.to_json(json_file, orient="records", indent=2)
print(f"Results saved to: {json_file}")

# %% [markdown]
# ## Appendix: Raw Metrics Command
#
# To manually check metrics from terminal:
# ```bash
# curl http://localhost:8000/metrics | grep vllm
# ```
#
# Key metrics to look for:
# - `vllm:time_to_first_token_seconds_sum/count` → TTFT
# - `vllm:request_time_per_output_token_seconds_sum/count` → TPOT
# - `vllm:e2e_request_latency_seconds_sum/count` → E2E
# - `vllm:request_prefill_time_seconds_sum` → Prefill time
# - `vllm:request_decode_time_seconds_sum` → Decode time
