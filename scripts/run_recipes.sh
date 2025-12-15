#!/bin/bash
# ============================================================================
# Training Recipes Runner Script
# ============================================================================
#
# This script runs training recipes for Qwen2-VL nutrition table detection.
# Each recipe requires 2 GPUs (model parallelism via device_map="balanced").
#
# Usage:
#   ./scripts/run_recipes.sh [recipe] [gpu_ids]
#   ./scripts/run_recipes.sh all        # Run all recipes sequentially
#   ./scripts/run_recipes.sh parallel   # Run r1 and r4 in parallel (4 GPUs)
#
# Examples:
#   ./scripts/run_recipes.sh r1 0,1     # Run r1-llm-only on GPUs 0,1
#   ./scripts/run_recipes.sh r2 2,3     # Run r2-vision-only on GPUs 2,3
#   ./scripts/run_recipes.sh r3 0,1     # Run r3-two-stage on GPUs 0,1
#   ./scripts/run_recipes.sh r4 0,1     # Run r4-joint on GPUs 0,1
#
# Output Paths:
#   r1-llm-only    -> /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r1-llm-only
#   r2-vision-only -> /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r2-vision-only
#   r3-two-stage   -> /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage1 (stage 1)
#                  -> /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r3-stage2 (stage 2)
#   r4-joint       -> /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint
# ============================================================================

set -e  # Exit on error

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_Qwen2VL_object_detection

# Change to project directory
cd /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection

# Default settings
EPOCHS=3
MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [recipe] [gpu_ids]"
    echo ""
    echo "Recipes:"
    echo "  r1        Run r1-llm-only (LLM QLoRA, vision frozen)"
    echo "  r2        Run r2-vision-only (Vision full fine-tuning, LLM frozen)"
    echo "  r3        Run r3-two-stage (Vision first, then LLM)"
    echo "  r4        Run r4-joint (Vision LoRA + LLM LoRA)"
    echo "  all       Run all recipes sequentially"
    echo "  parallel  Run r1 and r4 in parallel (requires 4 GPUs)"
    echo ""
    echo "Examples:"
    echo "  $0 r1 0,1     # Run r1 on GPUs 0,1"
    echo "  $0 r2 2,3     # Run r2 on GPUs 2,3"
    echo "  $0 all        # Run all recipes sequentially (auto GPU)"
    echo ""
}

run_recipe() {
    local recipe=$1
    local gpus=$2

    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Running recipe: ${recipe}${NC}"
    echo -e "${GREEN}GPUs: ${gpus}${NC}"
    echo -e "${GREEN}Epochs: ${EPOCHS}${NC}"
    echo -e "${GREEN}============================================${NC}"

    if [ -n "$gpus" ]; then
        python scripts/train_recipe.py --recipe "$recipe" --gpu "$gpus" --epochs "$EPOCHS"
    else
        python scripts/train_recipe.py --recipe "$recipe" --epochs "$EPOCHS"
    fi

    echo -e "${GREEN}Completed: ${recipe}${NC}"
}

# Main logic
case "${1:-}" in
    r1)
        run_recipe "r1-llm-only" "${2:-}"
        ;;
    r2)
        run_recipe "r2-vision-only" "${2:-}"
        ;;
    r3)
        run_recipe "r3-two-stage" "${2:-}"
        ;;
    r4)
        run_recipe "r4-joint" "${2:-}"
        ;;
    all)
        echo -e "${YELLOW}Running all recipes sequentially...${NC}"
        run_recipe "r1-llm-only" "${2:-}"
        run_recipe "r2-vision-only" "${2:-}"
        run_recipe "r3-two-stage" "${2:-}"
        run_recipe "r4-joint" "${2:-}"
        echo -e "${GREEN}All recipes completed!${NC}"
        ;;
    parallel)
        echo -e "${YELLOW}Running r1 and r4 in parallel (requires 4 GPUs)...${NC}"
        echo -e "${YELLOW}r1-llm-only on GPUs 0,1${NC}"
        echo -e "${YELLOW}r4-joint on GPUs 2,3${NC}"
        python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1 --epochs "$EPOCHS" &
        PID1=$!
        python scripts/train_recipe.py --recipe r4-joint --gpu 2,3 --epochs "$EPOCHS" &
        PID2=$!
        wait $PID1 $PID2
        echo -e "${GREEN}Parallel training completed!${NC}"
        ;;
    -h|--help|"")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown recipe: $1${NC}"
        print_usage
        exit 1
        ;;
esac
