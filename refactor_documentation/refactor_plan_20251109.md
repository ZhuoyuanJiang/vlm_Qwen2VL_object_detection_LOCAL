# Refactoring Plan - November 9, 2025

## Context
The main training script (`fine_tuning_vlm_for_object_detection_trl.py`) is 4,479 lines long with approximately:
- 39% exploratory data analysis
- 16% debugging code
- 10% custom collators (multiple versions)
- Multiple experimental approaches kept in the same file

**Goal:** Refactor to industry-standard structure while maintaining educational value for students.

---

## Planning Questions & Answers

### Q1: Which data collator version is the working/final one you want to keep?
**Answer:** "I want to keep multiple because I am developing scripts for students so they understand the difference between different collate functions difference. For now, let's try separate the dataset analysis before "# Understand Model" cell, so we can first make the .py file into more files to reduce the length of a single file."

### Q2: For the two training approaches in your file (train on all tokens vs train only on assistant responses), which do you want?
**Answer:** "Not sure yet"

### Q3: How should the EDA/exploratory analysis notebooks be organized?
**Answer:** "Your recommendation"

### Q4: Do you want test files for the modularized components?
**Answer:** "Only critical components"

---

## Strategy

### Main Notebook Philosophy
- **Keep comprehensive end-to-end demo** showing the full workflow
- Use the FINAL/BEST approach in the main notebook
- Add references to detailed exploration notebooks
- Students can see the complete pipeline in one place
- Later may condense sections, but keep the full flow visible

### Separate EDA Notebooks
- Detailed explorations for learning
- Alternative approaches for comparison
- Deep-dive analysis for students

---

## Phase 1 Implementation (Today)

### Step 0: Document the Plan ✓
Create `refactor_plan_20251109.md` (this file)

### Step 1: Create Directory Structure
Create `notebooks/` folder for educational materials

### Step 2: Create `01_dataset_exploration.ipynb`
Extract dataset EDA section (lines 243-784, ~541 lines) containing:
- Load OpenFoodFacts nutrition-table-detection dataset
- Dataset structure inspection (features, splits, samples)
- Visual analysis:
  - Image grid with bounding boxes (6 examples)
  - Image size distribution histograms
  - Bounding box count per image
  - Bounding box area coverage analysis
- Bounding box characteristics:
  - Aspect ratio distributions
  - Center position heatmaps
  - Size distributions
- Category/label analysis
- Anchor configuration recommendations
- Key insights for training

**Purpose:** Students get detailed understanding of dataset characteristics before training.

### Step 3: Update Main Script/Notebook
**Important:** DO NOT remove the EDA section from main notebook yet.

Instead:
- Add a markdown cell BEFORE the dataset EDA section with a note:
  ```
  # Dataset Analysis

  Note: For detailed dataset exploration and analysis, see `notebooks/01_dataset_exploration.ipynb`.
  Below shows a streamlined version sufficient for understanding the training pipeline.
  ```
- Keep the main notebook comprehensive (end-to-end demo)
- Decision on condensing deferred to later

### Step 4: Verify Setup
- Ensure jupytext sync works properly
- Test notebook creation and formatting
- Verify imports and paths work in the new notebook

---

## Proposed Professional Structure (Industry Standard)

### Full Directory Structure
```
vlm_Qwen2VL_object_detection/
│
├── notebooks/                                    # 📊 Educational & Exploratory
│   ├── 01_dataset_exploration.ipynb            # ✅ COMPLETED - Dataset EDA
│   ├── 02_model_understanding.ipynb            # ✅ COMPLETED - Model testing & inference experiments
│   ├── 03_verify_refactored_modules.ipynb      # ✅ COMPLETED - Verify extracted modules (temporary)
│   ├── 03_training_comparison.ipynb            # FUTURE - Compare different training approaches
│   └── 04_evaluation_analysis.ipynb            # FUTURE - Post-training analysis
│
├── src/                                         # 🔧 Core reusable package (renamed from vlm_modules)
│   ├── __init__.py
│   │
│   ├── data/                                    # Data processing module
│   │   ├── __init__.py
│   │   ├── dataset.py                           # Dataset loading & formatting functions
│   │   ├── collators.py                         # All 3 collator versions (educational)
│   │   └── preprocessing.py                     # Conversation formatting, vision info
│   │
│   ├── models/                                  # Model-related code
│   │   ├── __init__.py
│   │   ├── loader.py                            # Model loading with quantization
│   │   ├── inference.py                         # run_qwen2vl_inference, parse_qwen_bbox_output
│   │   └── lora.py                              # LoRA configuration helpers
│   │
│   ├── training/                                # Training infrastructure
│   │   ├── __init__.py
│   │   ├── callbacks.py                         # GradNormCallback, IoUEvalCallback, etc.
│   │   └── config.py                            # Training configuration dataclasses
│   │
│   └── utils/                                   # General utilities
│       ├── __init__.py
│       ├── gpu.py                               # find_available_gpus, GPU management
│       ├── visualization.py                     # visualize_bbox_on_image, plotting
│       └── metrics.py                           # IoU calculation, evaluation metrics
│
├── configs/                                     # ⚙️ Training configurations (YAML)
│   ├── base.yaml                                # Base config: model, data, common params
│   ├── training_all_tokens.yaml                 # Train on all tokens approach
│   ├── training_assistant_only.yaml             # Train only on assistant responses
│   └── lora_configs/
│       ├── aggressive.yaml                      # Higher rank LoRA
│       └── lightweight.yaml                     # Lower rank LoRA
│
├── scripts/                                     # 🚀 Executable scripts
│   ├── train.py                                 # Main training script (clean, ~150 lines)
│   ├── evaluate.py                              # Run evaluation on trained model
│   ├── inference.py                             # Run inference on new images
│   └── merge_lora.py                            # Merge LoRA adapters with base model
│
├── tests/                                       # 🧪 Unit tests
│   ├── test_collators.py                        # Test all 3 collator versions
│   ├── test_preprocessing.py                    # Test data formatting
│   └── test_inference.py                        # Test model inference
│
├── fine_tuning_vlm_for_object_detection_trl.ipynb   # 📓 Main comprehensive demo
├── fine_tuning_vlm_for_object_detection_trl.py      # Synced .py version
├── README.md
├── requirements.txt
└── refactor_plan_20251109.md                    # This document
```

### Key Structure Principles

**Why `src/` instead of `vlm_modules/`?**
- More standard in Python projects (follows PEP 420)
- Clearer that it's a package, not just modules
- Better for eventual PyPI packaging

**Why organize by function (`data/`, `models/`, `training/`)?**
- Intuitive: developers know where to find things
- Scalable: easy to add new components
- Testable: each module has clear responsibilities

**Why separate `scripts/` and `configs/`?**
- Clean CLI interface for running experiments
- Change hyperparameters without touching code
- Easy to version control different experiments
- Students can run experiments without coding

---

## Expected Outcome (Phase 1)

- ✅ Documentation file created with plan + Q&A
- ✅ New `notebooks/` directory for educational materials
- ✅ Detailed dataset exploration notebook for students
- ✅ Main notebook remains comprehensive with reference note
- ✅ Clear learning path established

**File changes:**
- Main notebook: Keep comprehensive (add reference note only)
- New: `01_dataset_exploration.ipynb` (~541 lines of detailed EDA)
- New: `refactor_plan_20251109.md` (this document)

---

## Future Phases (Not Today)

### Phase 2: Extract Model Understanding
Create `02_model_understanding.ipynb` with:
- Model architecture exploration
- Tokenizer analysis
- Pre-trained model testing
- Inference examples
- Output parsing experiments

### Phase 3: Modularize Utilities ✅ MOSTLY COMPLETED (Session 2)
Extract reusable code to `src/`:
- ✅ `src/utils/gpu.py`: GPU management
- ✅ `src/models/inference.py`: Inference, parsing
- ✅ `src/utils/visualization.py`: Bbox visualization
- 🔄 `src/data/collators.py`: Collator classes (needs V1/V2/V3 extraction)
- 📝 TODO: `src/training/callbacks.py`: Custom callbacks
- 📝 TODO: `src/utils/metrics.py`: Evaluation metrics

### Phase 4: Extract Evaluation Analysis
Create `03_evaluation_analysis.ipynb` with:
- Post-training evaluation
- IoU calculations
- Visualization of predictions
- Performance analysis

### Phase 5: Add Critical Tests
Add tests for:
- Data collators (critical component)
- Data preprocessing pipeline (critical component)

### Phase 6: Clean Up
- Remove debug sections
- Clean commented-out experimental code
- Consolidate duplicate code
- Document remaining training alternatives

---

## Detailed Component Examples

### Example 1: `src/data/collators.py` - Educational Collator Versions

```python
"""
Data collators for Qwen2-VL fine-tuning.

Contains multiple implementations for educational purposes:
- Qwen2VLCollatorV1: Basic implementation
- Qwen2VLCollatorV2: Fixed version with proper masking
- Qwen2VLCollatorV3: Final optimized version

Students can compare implementations to understand the evolution.
"""

class Qwen2VLCollatorV1:
    """
    First version - demonstrates basic approach.

    Issues:
    - May not properly mask user tokens
    - Basic batching logic
    """
    def __init__(self, processor, ...):
        self.processor = processor
        ...

    def __call__(self, examples):
        # Implementation
        ...

class Qwen2VLCollatorV2:
    """
    Improved version - fixed label masking.

    Improvements:
    - Properly masks user tokens when train_mode='assistant_only'
    - Better handling of vision tokens
    """
    ...

class Qwen2VLCollatorV3:
    """
    Final version - optimized for performance.

    Features:
    - Efficient batching
    - Configurable masking strategies
    - Better error handling
    """
    ...
```

### Example 2: `configs/training_assistant_only.yaml` - Clean Configuration

Separate hyperparameters from code using YAML files:
- **Model settings**: name, quantization (nf4), flash attention
- **Data settings**: dataset ID, splits, masking strategy
- **LoRA settings**: rank, alpha, target modules, dropout
- **Training settings**: batch size, learning rate, epochs, scheduler
- **Callbacks**: grad_norm, iou_eval, console_log
- **GPU config**: dual GPU, max GPUs

**Benefit**: Change experiments without touching code, easy to version control.

### Example 3: `scripts/train.py` - Clean Training Script

```python
#!/usr/bin/env python3
"""
Training script for Qwen2-VL nutrition table detection.

Usage:
    python scripts/train.py --config configs/training_assistant_only.yaml
    python scripts/train.py --config configs/training_all_tokens.yaml --output-dir ./custom
"""

import argparse
from pathlib import Path
import yaml

from src.utils.gpu import setup_gpu
from src.data.dataset import load_and_prepare_dataset
from src.data.collators import Qwen2VLCollatorV3  # Use final version
from src.models.loader import load_quantized_model_and_processor
from src.models.lora import setup_lora
from src.training.callbacks import get_callbacks
from src.training.config import create_training_config

from trl import SFTTrainer


def main(config_path: str, output_dir: str = None):
    """Main training function - only ~50 lines of actual logic!"""

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup components (each function is 5-20 lines, well-tested)
    setup_gpu(config['gpu'])
    model, processor = load_quantized_model_and_processor(config['model'])
    model = setup_lora(model, config['lora'])
    train_ds, eval_ds = load_and_prepare_dataset(config['data'], processor)
    collator = Qwen2VLCollatorV3(processor, config['data']['mask_prompt'])
    training_args = create_training_config(config['training'])
    callbacks = get_callbacks(config['callbacks'])

    # Create trainer and train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    print(f"🚀 Starting training: {config_path}")
    trainer.train()

    # Save
    trainer.save_model()
    processor.save_pretrained(config['training']['output_dir'])
    print(f"✅ Complete! Saved to {config['training']['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    main(args.config, args.output_dir)
```

---

## Key Benefits of This Structure

### ✅ For You (Research & Development)

**Easy Experimentation**
```bash
# Try different approaches without changing code
python scripts/train.py --config configs/training_assistant_only.yaml
python scripts/train.py --config configs/training_all_tokens.yaml
python scripts/train.py --config configs/experiments/high_lr.yaml
```

**Modular Debugging**
```python
# Test just the collator
from src.data.collators import Qwen2VLCollatorV3
collator = Qwen2VLCollatorV3(processor)
# Debug with sample data...

# Test just the preprocessing
from src.data.preprocessing import convert_to_conversation_format
result = convert_to_conversation_format(example)
# Inspect result...
```

**Version Control Friendly**
- Change one module without touching others
- Clear git diffs show what changed
- Easy to review and understand changes

### ✅ For Students (Educational)

**Clear Learning Path**
1. Start with `notebooks/01_dataset_exploration.ipynb` - understand data
2. Read `src/data/collators.py` - see 3 versions, understand evolution
3. Look at `configs/` - see how hyperparameters affect training
4. Run experiments with `scripts/train.py` - hands-on learning

**Compare Approaches**
```python
# Students can easily see differences
from src.data.collators import Qwen2VLCollatorV1, Qwen2VLCollatorV2, Qwen2VLCollatorV3

# Compare outputs
v1_output = Qwen2VLCollatorV1(processor)(examples)
v2_output = Qwen2VLCollatorV2(processor)(examples)
v3_output = Qwen2VLCollatorV3(processor)(examples)

# Analyze differences...
```

### ✅ For Production

**CLI Interface**
```bash
# Run training from command line
python scripts/train.py --config configs/training_assistant_only.yaml

# Evaluate model
python scripts/evaluate.py --model ./qwen2vl-nutrition-detection-lora

# Run inference
python scripts/inference.py \
    --model ./qwen2vl-nutrition-detection-lora \
    --image path/to/product.jpg
```

**Package Installation**
```bash
# Install as a package
pip install -e .

# Use in other projects
from src.models.inference import run_qwen2vl_inference
from src.utils.metrics import calculate_iou
```

**Testing**
```bash
# Run all tests
pytest tests/

# Test specific component
pytest tests/test_collators.py -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Comparison: Before vs After

### Before (Current)
```bash
# One massive file
fine_tuning_vlm_for_object_detection_trl.py  # 4,479 lines

# To train:
# - Open notebook or run .py file
# - Edit hyperparameters directly in code
# - Hope you don't break something

# To experiment:
# - Copy-paste entire section
# - Modify and hope it works
# - Lose track of what changed

# To debug:
# - Read through thousands of lines
# - Hard to isolate issues
# - Difficult to test components
```

### After (Professional)
```bash
# Organized structure
notebooks/          # 4 focused notebooks
src/               # 15 clean, tested modules (~200 lines each)
configs/           # 5+ YAML files (hyperparameters separate)
scripts/           # 4 executable scripts (~150 lines each)
tests/             # 3 test files

# To train:
python scripts/train.py --config configs/training_assistant_only.yaml

# To experiment:
# - Copy config file
# - Change hyperparameters
# - Run with new config
# - Track experiments with git

# To debug:
# - Test individual modules: pytest tests/test_collators.py
# - Import and inspect: from src.data import collators
# - Clear separation of concerns
```

---

## Implementation Recommendation

**IMPORTANT: Fix training bugs first, then refactor!**

Why?
1. **Easier to debug one file**: All code in one place for now
2. **Once it works**: Extract working components into clean modules
3. **Test-driven**: Write tests as you extract to ensure nothing breaks
4. **Educational**: Students can see both versions (monolithic vs modular)

**Today's achievement is valuable**: Dataset EDA extracted to notebook (12% reduction, clearer structure)

**When to do full refactoring**: After validation loss decreases correctly and training works

**How long it takes**: 1-2 hours for experienced developer, guided by this plan

---

## Notes

- Keep multiple collator versions for educational purposes
- Keep both training approaches (all-tokens vs assistant-only) for comparison
- Maintain comprehensive main notebook as end-to-end reference
- Separate detailed explorations into focused notebooks
- Prioritize clarity and learning value over minimal line count
- This structure is industry-standard for ML research and production
