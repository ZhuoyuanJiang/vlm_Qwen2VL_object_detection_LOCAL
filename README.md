# Qwen2-VL Fine-tuning for Nutrition Table Detection

Fine-tuning Qwen2-VL-7B model for detecting nutrition tables in food packaging images.

> **Start Here**: Read [`fine_tuning_vlm_for_object_detection_trl.ipynb`](fine_tuning_vlm_for_object_detection_trl.ipynb) first. This notebook demonstrates the **entire pipeline** from data loading to training to evaluation to deployment. It contains the core logic that all other scripts are built upon. This is the recommended entry point for interviewers, hiring managers, and anyone who wants to learn about this project.

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda activate vlm_Qwen2VL_object_detection
cd ~/projects/vlm_Qwen2VL_object_detection
```

### 2. HuggingFace Token
Token configured in `~/.bashrc` - works automatically for all projects.

### 3. Run Training

**Option A: Recipe-based training (recommended for production)**
```bash
# Run best-performing recipe (r4-joint) on GPUs 0,1
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1

# With logging
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1 2>&1 | tee r4-joint.log
```

**Option B: Interactive notebook (recommended for learning)**
```bash
jupyter notebook fine_tuning_vlm_for_object_detection_trl.ipynb
# Open the notebook and run all cells for the complete pipeline
```

## 📦 Environment & Requirements

| Component | Version/File |
|-----------|--------------|
| **Python** | 3.10.18 |
| **Portable env** | `environment.yml` (no builds, cross-platform) |
| **Exact snapshot** | `environment.lock.yml` (with builds, reproducible) |
| **Pip packages** | `requirements.txt` |

### Key Dependencies
- `transformers` (git HEAD) - HuggingFace transformers
- `trl` (git HEAD, ~0.22.0.dev0) - TRL training library with `completion_only_loss`
- `peft==0.17.1` - LoRA/QLoRA
- `bitsandbytes==0.47.0` - 4-bit quantization
- `torch==2.4.1+cu121` - PyTorch with CUDA 12.1

### Recreate Environment
```bash
# From portable export (recommended)
conda env create -f environment.yml

# From exact snapshot (same platform only)
conda env create -f environment.lock.yml
```

## ⚠️ TRL Version Note

**This project uses TRL ~0.22.0.dev0** (git-based) with the new `completion_only_loss` parameter:

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    completion_only_loss=True,  # Train only on assistant responses
    # other args...
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    # No need for DataCollatorForCompletionOnlyLM anymore
)
```

> **Note**: The old `DataCollatorForCompletionOnlyLM` from TRL 0.10.1 is no longer needed. The TRL team integrated completion-only functionality directly into the trainer.

## 📁 Project Files

- `fine_tuning_vlm_for_object_detection_trl.ipynb` - **Comprehensive educational notebook** covering the full pipeline: data loading, preprocessing, model setup, training, evaluation, and deployment
- `fine_tuning_vlm_for_object_detection_trl.py` - Python script (synced with notebook via Jupytext)
- `scripts/train_recipe.py` - **Production training script** for running different recipes
- `notebooks/` - **Topic-specific educational notebooks** (EDA, model understanding, evaluation, etc.)
- `requirements.txt` - Pip dependencies
- `environment.yml` - Conda environment (portable)
- `environment.lock.yml` - Conda environment (exact)

## 🔁 Notebook ⇄ Script Sync (Jupytext)

> **Note**: Replace paths below with your project location.

- One-time pairing (sets formats on the notebook):
  ```bash
  jupytext --set-formats ipynb,py:percent /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/fine_tuning_vlm_for_object_detection_trl.ipynb
  ```

- Ongoing sync (both directions; updates the older file from the newer one):
  ```bash
  jupytext --sync /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/fine_tuning_vlm_for_object_detection_trl.ipynb | cat
  ```

- Directional sync (optional control):
  - ipynb → py
    ```bash
    jupytext --to py:percent /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/fine_tuning_vlm_for_object_detection_trl.ipynb
    ```
  - py → ipynb
    ```bash
    jupytext --to ipynb /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/fine_tuning_vlm_for_object_detection_trl.py
    ```

## 🏗️ Project Structure - Overview

```
vlm_Qwen2VL_object_detection/
├── src/                          # Modular source code
│   ├── data/                     # Dataset preparation & collation
│   ├── models/                   # Model loading & configuration
│   ├── training/                 # Training utilities & evaluation
│   └── utils/                    # GPU management & visualization
├── scripts/                      # Training scripts (see below for details)
│   ├── train.py                  # Simple single-config training
│   ├── train_recipe.py           # Flexible recipe-based training (recommended)
│   └── run_recipes.sh            # Bash wrapper for train_recipe.py
├── tests/                        # Test suite
│   ├── test_data_format_before_chat_template.py
│   └── test_golden_output.py
├── notebooks/                    # Educational notebooks (topic-specific deep dives)
│   ├── 01_dataset_exploration.ipynb      # EDA and data understanding
│   ├── 02_model_understanding.ipynb      # Qwen2-VL architecture exploration
│   ├── 03_data_preprocessing.ipynb       # Data format and preprocessing
│   ├── 04_evaluation_analysis.ipynb      # Model evaluation and metrics
│   ├── 05_debug_dtype_issue.ipynb        # Debugging dtype/device issues
│   └── 06_quantization_and_trainable_params.ipynb  # Quantization deep dive
└── refactor_documentation/       # Development history (23 sessions)
```

## 📂 Project Structure - Detailed (src/)

### `src/data/`
| File | Description |
|------|-------------|
| `dataset.py` | Convert raw data to conversation format with image placeholders |
| `collators.py` | Batch collation: `AllTokensCollator`, `AssistantOnlyCollator` |

### `src/models/`
| File | Description |
|------|-------------|
| `loader.py` | Load Qwen2-VL with 4-bit quantization and device_map |
| `lora.py` | LoRA config (r=64, alpha=128) and target modules |
| `inference.py` | Single image inference and bbox output parsing |

### `src/training/`
| File | Description |
|------|-------------|
| `config.py` | SFTConfig creation and training variants |
| `evaluation.py` | IoU-based evaluation metrics |
| `callbacks.py` | *(LEGACY)* Flash attention debug callbacks, not currently used |

### `src/utils/`
| File | Description |
|------|-------------|
| `gpu.py` | Auto-detect GPUs, setup CUDA_VISIBLE_DEVICES |
| `visualization.py` | Plotting and visualization helpers |
| `debug.py` | Debugging utilities |

## 📜 Scripts Folder

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_recipe.py` | Flexible recipe-based training with CLI args | **Recommended** for all training runs |
| `train.py` | Simple training script (~150 lines) with hardcoded config | Learning the codebase, quick experiments |
| `run_recipes.sh` | Bash wrapper with simpler syntax | Convenience for running recipes |

### `train_recipe.py` (Recommended)
Full-featured training with command-line arguments:
```bash
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1 --epochs 3 --no-wandb
```
- Supports all 4 recipes (r1, r2, r3, r4)
- GPU selection via `--gpu`
- Configurable epochs, W&B toggle

### `train.py` (AllTokensCollator)
Minimal script using `AllTokensCollator`:
```bash
python scripts/train.py
```
- Uses `AllTokensCollator`:
  - Masks: pad tokens + vision tokens (`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`)
  - Trains on: all text tokens (system + user + assistant)
- `train_recipe.py` uses `AssistantOnlyCollator`: Trains only on object category + bbox coordinates
- See main notebook for collator comparison experiments

### `run_recipes.sh` (Convenience)
Bash wrapper with shorter syntax:
```bash
./scripts/run_recipes.sh r4 0,1      # Run r4-joint on GPUs 0,1
./scripts/run_recipes.sh all         # Run all recipes sequentially
./scripts/run_recipes.sh parallel    # Run r1 and r4 in parallel (4 GPUs)
```

## 🧪 Training Recipes

**Entry point**: `scripts/train_recipe.py`

Four training approaches have been implemented and evaluated:

| Recipe | Strategy | Mean IoU | Detection Rate | IoU>0.5 | IoU>0.7 |
|--------|----------|----------|----------------|---------|---------|
| Base (no fine-tuning) | - | 0.0981 | 18.00% | 10.00% | 6.00% |
| r1-llm-only | 4-bit + LoRA on LLM | 0.8349 | 100.00% | 86.00% | 82.00% |
| r2-vision-only | bf16 full fine-tune vision | 0.8330 | 100.00% | 88.00% | 82.00% |
| r3-two-stage | Vision first, then LLM LoRA | 0.8366 | 100.00% | 90.00% | 80.00% |
| **r4-joint** | **4-bit + LoRA on both** | **0.8636** | **100.00%** | **92.00%** | **88.00%** |

**Best performer: r4-joint** with 0.8636 mean IoU (+780% improvement over base model)

### Training Commands

```bash
# Run r1-llm-only on GPUs 0,1
python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1

# Run r2-vision-only on GPUs 2,3
python scripts/train_recipe.py --recipe r2-vision-only --gpu 2,3

# Run r4-joint on GPUs 4,5
python scripts/train_recipe.py --recipe r4-joint --gpu 4,5

# Run r3-two-stage (runs both stages sequentially)
python scripts/train_recipe.py --recipe r3-two-stage --gpu 6,7

# Disable W&B logging
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1 --no-wandb

# With logging to file (using tee)
python scripts/train_recipe.py --recipe r4-joint --gpu 0,1 2>&1 | tee r4-joint_$(date +%Y%m%d_%H%M%S).log
```

### Run Multiple Recipes in Parallel
```bash
# Terminal 1
python scripts/train_recipe.py --recipe r1-llm-only --gpu 0,1

# Terminal 2
python scripts/train_recipe.py --recipe r2-vision-only --gpu 2,3

# Terminal 3
python scripts/train_recipe.py --recipe r4-joint --gpu 4,5
```

## 📈 Performance Results

Based on evaluation of 50 test samples:

| Metric | r4-joint (Best) |
|--------|-----------------|
| **Mean IoU** | 0.8636 |
| **Median IoU** | 0.89 |
| **Detection Rate** | 100% |
| **IoU > 0.5** | 92% |
| **IoU > 0.7** | 88% |

## 🔧 Configuration

### Model
- **Base**: Qwen2-VL-7B-Instruct
- **Quantization**: 4-bit NF4 with BitsAndBytes
- **Fine-tuning**: LoRA (r=64, alpha=128)
- **Training**: Completion-only loss on assistant responses

### Hardware
- **Tested on**: RTX 6000 Ada (47.50 GB VRAM)
- **Minimum Required**: 24GB VRAM
- **Multi-GPU**: Supported via device_map="balanced"

### Key Hyperparameters
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 1e-5 to 2e-5 with cosine schedule
- **Epochs**: 3
- **LoRA Rank**: 64 (higher for vision tasks)
- **LoRA Alpha**: 128
- **Max Length**: 2048 tokens

### Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking.
```bash
wandb login
```

## 📊 Dataset

Using nutrition table detection dataset from HuggingFace:
- **Source**: OpenFoodFacts nutrition-table-detection
- **Training samples**: ~1083
- **Task**: Detect bounding boxes of nutrition tables
- **Format**: Conversation-style with user prompts and assistant responses

## 📁 Training Output Structure

Running `scripts/train_recipe.py` creates outputs in `/ssd1/zhuoyuan/vlm_outputs/`:

```
/ssd1/zhuoyuan/vlm_outputs/
├── qwen2vl-nutrition-detection-r1-llm-only/      # r1 recipe output
│   ├── checkpoint-*/                              # Intermediate checkpoints
│   ├── adapter_model.safetensors                  # LoRA adapter weights
│   ├── adapter_config.json                        # LoRA config
│   └── processor files...
├── qwen2vl-nutrition-detection-r2-vision-only/   # r2 recipe output (full model)
├── qwen2vl-nutrition-detection-r3-stage1/        # r3 Stage 1: vision training
├── qwen2vl-nutrition-detection-r3-stage2/        # r3 Stage 2: LLM from stage1
├── qwen2vl-nutrition-detection-r4-joint/         # r4 recipe output
├── qwen2vl-nutrition-detection-lora/             # Legacy: main notebook output
├── qwen2vl-nutrition-detection-merged/           # Optional: merged full model
└── logs/                                          # TensorBoard logs

/ssd1/zhuoyuan/hf_cache/                          # Model downloads (cached)
```

**Note**: r3-two-stage creates two outputs (`-r3-stage1` and `-r3-stage2`) because it runs sequentially.

**Storage Strategy**: All large files on SSD to preserve home directory quota (~100GB).

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | Reduce batch_size or enable gradient_checkpointing |
| **Dataset Access** | HF token should be set in ~/.bashrc |
| **dtype mismatch error** | Check device_map and DataParallel conflicts (see Session 20 docs) |
| **Evaluation showing 0% detection** | Add `gc.collect()` and `torch.cuda.empty_cache()` before loading models |

## 📄 License

For educational purposes. Check individual licenses for Qwen2-VL model and datasets.
