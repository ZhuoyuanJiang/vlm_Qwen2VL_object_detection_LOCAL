#!/bin/bash
# Entrypoint for Qwen2-VL Triton Inference Server
#
# Usage:
#   docker run ... <image> gptq     # GPTQ INT4 model (default)
#   docker run ... <image> bf16     # BF16 full-precision model
#   docker run ... <image> both     # Both models (requires 2+ GPUs)

set -e

MODEL="${1:-gptq}"
REPO_DIR="/models/triton_repo"
mkdir -p "$REPO_DIR"

case "$MODEL" in
    gptq)
        cp -r /opt/triton_configs/qwen2vl_nutrition_gptq_int4 "$REPO_DIR/"
        echo "=== Starting Triton with GPTQ INT4 model ==="
        echo "Mount weights at: /models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"
        ;;
    bf16)
        cp -r /opt/triton_configs/qwen2vl_nutrition_bf16 "$REPO_DIR/"
        echo "=== Starting Triton with BF16 model ==="
        echo "Mount weights at: /models/qwen2vl-nutrition-detection-r4-joint-merged"
        ;;
    both)
        cp -r /opt/triton_configs/qwen2vl_nutrition_gptq_int4 "$REPO_DIR/"
        cp -r /opt/triton_configs/qwen2vl_nutrition_bf16 "$REPO_DIR/"
        echo "=== Starting Triton with both models ==="
        echo "Mount GPTQ weights at: /models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"
        echo "Mount BF16 weights at:  /models/qwen2vl-nutrition-detection-r4-joint-merged"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: docker run ... <image> [gptq|bf16|both]"
        exit 1
        ;;
esac

echo ""
echo "HTTP:    http://localhost:8000"
echo "gRPC:    localhost:8001"
echo "Metrics: http://localhost:8002/metrics"
echo ""

exec tritonserver --model-repository="$REPO_DIR"
