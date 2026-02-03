#!/bin/bash
# =============================================================================
# Triton Deployment Script for Qwen2-VL Nutrition Detection
# =============================================================================
#
# This script starts the Triton Inference Server with vLLM backend.
# It mounts both BF16 and GPTQ INT4 models for A/B testing.
#
# Prerequisites:
#   - Docker with GPU support (nvidia-docker or --gpus flag)
#   - Model weights transferred to cloud server
#   - Triton model repository with config files
#
# Usage:
#   ./deploy_triton.sh                    # Start with defaults
#   ./deploy_triton.sh --single gptq      # Single model mode (GPTQ only)
#   ./deploy_triton.sh --single bf16      # Single model mode (BF16 only)
#   ./deploy_triton.sh --detach           # Run in background
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration - UPDATE THESE PATHS FOR YOUR CLOUD SERVER
# -----------------------------------------------------------------------------

# Model weights paths (on cloud server host)
BF16_MODEL_PATH="/models/qwen2vl-nutrition-detection-r4-joint-merged"
GPTQ_MODEL_PATH="/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"

# Triton model repository path (on cloud server host)
TRITON_REPO_PATH="/workspace/triton_model_repository"

# Docker image
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3"

# Ports
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002

# Shared memory size (important for vLLM)
SHM_SIZE="4G"

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------

DETACH=""
SINGLE_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --detach|-d)
            DETACH="-d"
            shift
            ;;
        --single)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --detach, -d       Run in background (detached mode)"
            echo "  --single MODEL     Run single model only (bf16 or gptq)"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Build volume mounts based on mode
# -----------------------------------------------------------------------------

VOLUME_MOUNTS=""

if [[ "$SINGLE_MODEL" == "bf16" ]]; then
    echo "Starting Triton with BF16 model only..."
    VOLUME_MOUNTS="-v ${BF16_MODEL_PATH}:${BF16_MODEL_PATH}:ro"
elif [[ "$SINGLE_MODEL" == "gptq" ]]; then
    echo "Starting Triton with GPTQ INT4 model only..."
    VOLUME_MOUNTS="-v ${GPTQ_MODEL_PATH}:${GPTQ_MODEL_PATH}:ro"
else
    echo "Starting Triton with both models (BF16 + GPTQ INT4)..."
    VOLUME_MOUNTS="-v ${BF16_MODEL_PATH}:${BF16_MODEL_PATH}:ro -v ${GPTQ_MODEL_PATH}:${GPTQ_MODEL_PATH}:ro"
fi

# Add triton repo mount
VOLUME_MOUNTS="${VOLUME_MOUNTS} -v ${TRITON_REPO_PATH}:/models/triton_repo"

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

echo ""
echo "=== Pre-flight Checks ==="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi
echo "✓ Docker available"

# Check if nvidia-docker/GPU support is available
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "ERROR: Docker GPU support not available. Ensure nvidia-docker is installed."
    exit 1
fi
echo "✓ Docker GPU support available"

# Check if model paths exist
if [[ "$SINGLE_MODEL" != "gptq" ]] && [[ ! -d "$BF16_MODEL_PATH" ]]; then
    echo "WARNING: BF16 model path not found: $BF16_MODEL_PATH"
fi

if [[ "$SINGLE_MODEL" != "bf16" ]] && [[ ! -d "$GPTQ_MODEL_PATH" ]]; then
    echo "WARNING: GPTQ model path not found: $GPTQ_MODEL_PATH"
fi

if [[ ! -d "$TRITON_REPO_PATH" ]]; then
    echo "ERROR: Triton repo path not found: $TRITON_REPO_PATH"
    exit 1
fi
echo "✓ Triton repo found"

# -----------------------------------------------------------------------------
# Start Triton Server
# -----------------------------------------------------------------------------

echo ""
echo "=== Starting Triton Server ==="
echo "HTTP:    http://localhost:${HTTP_PORT}"
echo "gRPC:    localhost:${GRPC_PORT}"
echo "Metrics: http://localhost:${METRICS_PORT}/metrics"
echo ""

docker run --gpus all --rm ${DETACH} -it \
    --shm-size=${SHM_SIZE} \
    -p ${HTTP_PORT}:8000 \
    -p ${GRPC_PORT}:8001 \
    -p ${METRICS_PORT}:8002 \
    ${VOLUME_MOUNTS} \
    ${TRITON_IMAGE} \
    tritonserver --model-repository=/models/triton_repo

# If running in detached mode, show how to check status
if [[ -n "$DETACH" ]]; then
    echo ""
    echo "Triton server started in background."
    echo ""
    echo "To check status:"
    echo "  curl http://localhost:${HTTP_PORT}/v2/health/live"
    echo ""
    echo "To view logs:"
    echo "  docker logs -f \$(docker ps -q --filter ancestor=${TRITON_IMAGE})"
    echo ""
    echo "To stop:"
    echo "  docker stop \$(docker ps -q --filter ancestor=${TRITON_IMAGE})"
fi
