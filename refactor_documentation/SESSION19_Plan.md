# Plan: Parallel Training Script for Recipes

**Date:** 2025-12-14
**Goal:** Create a standalone script that can run training recipes in parallel on different GPUs

---

## Background

The recipe framework was added to `fine_tuning_vlm_for_object_detection_trl.py` (Session 18), but it runs sequentially in a Jupyter notebook. The user wants to run multiple recipes in parallel on different GPUs.

**Existing Resources:**
- `scripts/train.py` - Existing training script using src/ modules
- `src/data/collators.py` - Contains `AllTokensCollator`, `AssistantOnlyCollator`
- `src/training/config.py` - Contains `create_training_config()`
- `src/models/loader.py` - Contains `load_quantized_model()`
- `src/models/lora.py` - Contains `create_lora_config()`
- `src/data/dataset.py` - Contains `convert_to_conversation_format()`
- `src/utils/gpu.py` - Contains `setup_gpus()`

---

## Implementation Plan

### Create `scripts/train_recipe.py`

A new standalone script that:
1. Accepts command-line arguments for recipe and GPU selection
2. Supports both `--gpu` flag and `CUDA_VISIBLE_DEVICES` environment variable
3. Imports from src/ modules (keeps code DRY)
4. Contains the recipe definitions (dataclass + RECIPES dict)
5. Can be run multiple times in parallel on different GPUs

### Script Structure

```
scripts/train_recipe.py
├── Argument parsing (--recipe, --gpu, --epochs)
├── GPU setup (before importing torch)
├── Recipe definitions (TrainingRecipe, RECIPES, target modules)
├── Recipe helper functions (load_model_for_recipe, apply_freeze_policy, build_lora_config)
├── Main execution (data loading, training, saving)
└── Entry point with usage examples
```

### Usage Examples

```bash
# Run r1-llm-only on GPU 0
python scripts/train_recipe.py --recipe r1-llm-only --gpu 0

# Run r2-vision-only on GPU 1 (using env var)
CUDA_VISIBLE_DEVICES=1 python scripts/train_recipe.py --recipe r2-vision-only

# Run multiple recipes in parallel (different terminals/screens)
# Terminal 1:
python scripts/train_recipe.py --recipe r1-llm-only --gpu 0

# Terminal 2:
python scripts/train_recipe.py --recipe r2-vision-only --gpu 1

# Terminal 3:
python scripts/train_recipe.py --recipe r4-joint --gpu 2
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--recipe` | Yes | - | Recipe key: r1-llm-only, r2-vision-only, r3-two-stage, r4-joint |
| `--gpu` | No | auto | GPU ID(s) to use (e.g., "0" or "0,1") |
| `--epochs` | No | 3 | Number of training epochs |
| `--checkpoint` | No | None | Path to load from (for two-stage) |
| `--model-id` | No | Qwen/Qwen2-VL-7B-Instruct | Model ID |
| `--no-wandb` | No | False | Disable W&B logging |

### Key Code Components

**1. GPU setup (must be before torch import):**
```python
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--recipe", required=True, choices=["r1-llm-only", "r2-vision-only", "r3-two-stage", "r4-joint"])
parser.add_argument("--gpu", type=str, default=None, help="GPU ID(s), e.g., '0' or '0,1'")
# ... more args
args = parser.parse_args()

# Set GPU BEFORE importing torch
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
```

**2. Recipe definitions (copy from main file):**
```python
from dataclasses import dataclass
from typing import Optional, List

LLM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VISION_LORA_TARGETS = ["qkv", "fc1", "fc2"]

@dataclass
class TrainingRecipe:
    name: str
    quantized: bool
    train_vision: str
    train_llm: str
    lora_target_modules: Optional[List[str]]
    learning_rate: float
    description: str

RECIPES = {
    "r1-llm-only": TrainingRecipe(...),
    "r2-vision-only": TrainingRecipe(...),
    "r3-two-stage": TrainingRecipe(...),
    "r4-joint": TrainingRecipe(...),
}
```

**3. Import from src/ modules:**
```python
from src.models.loader import load_quantized_model
from src.models.lora import create_lora_config
from src.training.config import create_training_config
from src.data.collators import AssistantOnlyCollator
from src.data.dataset import convert_to_conversation_format, _has_image
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/train_recipe.py` | **CREATE** | New standalone recipe training script |
| `scripts/train.py` | No change | Keep existing script as-is |

---

## Two-Stage Recipe Handling

For r3-two-stage, the script will:
1. If `--recipe r3-two-stage` is passed without `--checkpoint`:
   - First run r2-vision-only, save checkpoint
   - Then run r1-llm-only from that checkpoint
   - Both stages run sequentially on same GPU

2. Alternatively, user can run stages manually:
   ```bash
   # Stage 1 on GPU 0
   python scripts/train_recipe.py --recipe r2-vision-only --gpu 0
   # After completion, Stage 2 on GPU 0 (or any GPU)
   python scripts/train_recipe.py --recipe r1-llm-only --gpu 0 --checkpoint /path/to/stage1
   ```

---

## Important: Correct Parameter Path Patterns

When implementing `apply_freeze_policy`, use the correct parameter name patterns:

```python
# Parameter paths from model.named_parameters():
#   Vision: model.visual.blocks.0.attn.qkv.weight
#   LLM:    model.language_model.layers.0.self_attn.q_proj.weight

# CORRECT patterns for freeze/unfreeze:
if ".visual." in name:        # Vision encoder
if "language_model" in name:  # LLM backbone

# WRONG (bug fixed in Session 18):
if "model.layers" in name:    # Does NOT match "model.language_model.layers"!
```

---

## Validation Before Implementation

- [ ] Confirm GPU memory is sufficient for each recipe type
- [ ] Verify src/ modules are importable without issues
- [ ] Test that CUDA_VISIBLE_DEVICES works correctly for multi-GPU model sharding
