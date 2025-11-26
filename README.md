# Qwen2-VL Fine-tuning for Nutrition Table Detection

Fine-tuning Qwen2-VL-7B model for detecting nutrition tables in food packaging images.

## 🚀 Quick Start

### Environment Setup (✅ Already Complete)
```bash
conda activate vlm_Qwen2VL_object_detection
```

### HuggingFace Token (✅ Already Set)
Token configured in `~/.bashrc` - works automatically for all projects.

### Run Training
```bash
python fine_tuning_vlm_for_object_detection_trl.py
# Or use notebook for interactive development
jupyter notebook fine_tuning_vlm_for_object_detection_trl.ipynb
```

## ⚠️ Important: TRL Version Note

**This project uses TRL 0.10.1** (not latest) because we need `DataCollatorForCompletionOnlyLM`:

| TRL Version | DataCollatorForCompletionOnlyLM | Alternative Method |
|-------------|----------------------------------|-------------------|
| 0.10.1 (ours) | ✅ Available | - |
| 0.22.0 (latest) | ❌ Removed | Use `completion_only_loss=True` in SFTConfig |

**The TRL team integrated completion-only functionality directly into the trainer in v0.22:**

Old Way (TRL 0.10.1 - what we use):
```python
from trl import DataCollatorForCompletionOnlyLM
data_collator = DataCollatorForCompletionOnlyLM(
    response_template="### Response:",
    tokenizer=tokenizer
)
```

New Way (TRL 0.22+):
```python
from trl import SFTConfig
training_args = SFTConfig(
    completion_only_loss=True,  # Replaces DataCollatorForCompletionOnlyLM
    # other args...
)
```

### Install Correct TRL Version:
```bash
pip uninstall -y trl
pip install trl==0.10.1
```

## 📁 Project Files

- `fine_tuning_vlm_for_object_detection_trl.ipynb` - Main training notebook
- `fine_tuning_vlm_for_object_detection_trl.py` - Python script (synced with notebook)
- `PROJECT_ANALYSIS.md` - Detailed hyperparameter rationale and architecture
- `DETAILED_SESSION_LOG.md` - Complete setup history
- `requirements.txt` - Dependencies

## 🔁 Notebook ⇄ Script Sync (Jupytext)

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

- Optional repo config (so you don’t need --set-formats again):
  ```bash
  printf 'formats = "ipynb,py:percent"\n' > /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/.jupytext.toml
  ```

- Activate env only if your shell isn’t already in it:
  ```bash
  source /home/zhuoyuan/miniconda3/etc/profile.d/conda.sh && conda activate vlm_Qwen2VL_object_detection
  ```

## 🔧 Configuration

### Model
- **Base**: Qwen2-VL-7B-Instruct
- **Quantization**: 4-bit NF4 with BitsAndBytes
- **Fine-tuning**: LoRA (r=64, alpha=16)
- **Training**: Completion-only loss on assistant responses

### Hardware
- **Your GPU**: RTX 6000 Ada (47.50 GB VRAM) ✨
- **Minimum Required**: 24GB VRAM
- **Recommended**: With your GPU, use batch_size=8 for faster training

### Key Hyperparameters
- **Batch Size**: 2 per device (can increase to 8)
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 1e-5 with cosine schedule
- **Epochs**: 3
- **LoRA Rank**: 64 (higher for vision tasks)
- **Max Length**: 2048 tokens

See `PROJECT_ANALYSIS.md` for detailed explanation of every parameter choice.

### Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking.
When you first run training, you'll be prompted to login or create a free account.

To skip the prompt if you already have WandB:
```bash
wandb login
```

## 📊 Dataset

Using nutrition table detection dataset from HuggingFace:
- Training samples: ~1083
- Task: Detect bounding boxes of nutrition tables
- Format: Conversation-style with user prompts and assistant responses

## 🎯 Quick Start

1. **Activate Environment** (new terminal):
   ```bash
   conda activate vlm_Qwen2VL_object_detection
   cd ~/projects/vlm_Qwen2VL_object_detection
   ```

2. **Verify Setup**:
   ```bash
   python -c "from trl import DataCollatorForCompletionOnlyLM; print('✅ TRL 0.10.1 ready!')"
   ```

3. **Start Training**:
   ```bash
   # Full training (~3-6 hours)
   python fine_tuning_vlm_for_object_detection_trl.py
   
   # Or run in background
   nohup python fine_tuning_vlm_for_object_detection_trl.py > training.log 2>&1 &
   ```

## 📁 Training Output Structure (ALL ON SSD - Home stays <10MB!)

```
/ssd1/zhuoyuan/vlm_outputs/
├── qwen2vl-nutrition-detection-lora/   # Main output directory
│   ├── checkpoint-300/                 # Saved every 300 steps
│   ├── checkpoint-600/                 # Only keeps last 3
│   ├── checkpoint-900/                 # Auto-cleanup of old ones
│   └── [Final outputs]:
│       ├── adapter_model.safetensors   # Final LoRA (~200MB)
│       └── training_results.json       # Metrics
├── qwen2vl-nutrition-detection-merged/ # Optional: Full model
└── logs/                               # TensorBoard logs

/ssd1/zhuoyuan/hf_cache/               # Model downloads
```

**Storage**: ALL on SSD (383GB available). Home directory: <10MB only!

## 📈 Expected Performance

- **Training Time**: 3-6 hours
- **Memory Usage**: ~12GB with current settings
- **Expected mAP**: 85-95% on nutrition tables
- **Inference Speed**: ~200ms per image

## ✅ Current Status

- ✅ Environment setup complete
- ✅ All dependencies installed 
- ✅ TRL 0.10.1 installed (has DataCollatorForCompletionOnlyLM)
- ✅ HuggingFace token configured in ~/.bashrc
- ✅ GPU verified (RTX 6000 Ada, 47.50 GB VRAM)
- ✅ **READY FOR TRAINING!**

**Note on ~/.bashrc**: Your HF token loads automatically when opening a new terminal. No need to source it manually unless continuing the same terminal session.

## 📚 Documentation

- **Detailed Setup Log**: See `DETAILED_SESSION_LOG.md`
- **Architecture & Rationale**: See `PROJECT_ANALYSIS.md`
- **Original Guide**: See `WORKFLOW_GUIDE.md`

## 🔍 Troubleshooting

Common issues and solutions:
1. **Import Error**: Make sure TRL 0.10.1 is installed
2. **CUDA OOM**: Reduce batch_size or enable gradient_checkpointing
3. **Dataset Access**: HF token is already set in ~/.bashrc

## 📄 License

For educational purposes. Check individual licenses for Qwen2-VL model and datasets.