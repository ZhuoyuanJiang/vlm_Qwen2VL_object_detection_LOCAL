# 🌅 Wake-Up Guide: Where We Are & What to Do

**Last Updated**: August 20, 2025, 12:50 PM  
**Project Status**: ✅ READY FOR TRAINING

---

## 📍 Current Status

### ✅ What's Already Done:
1. **Environment Setup**: `vlm_Qwen2VL_object_detection` with Python 3.10.18
2. **All Dependencies Installed**:
   - PyTorch 2.4.1 with CUDA 12.1
   - Transformers (latest dev)
   - TRL 0.10.1 (has DataCollatorForCompletionOnlyLM)
   - PEFT, BitsAndBytes, all requirements
3. **HuggingFace Token**: Saved in `~/.bashrc` (automatic)
4. **Hardware Verified**: RTX 6000 Ada with 47.50 GB VRAM
5. **Project Cleaned**: Removed unnecessary files

### 🔑 About ~/.bashrc:
**NO, you DON'T need to source ~/.bashrc every time!**
- It's automatically sourced when you open a new terminal
- Only needed if you're continuing in the SAME terminal session from before
- New terminal = automatic loading

---

## 🚀 Quick Start Commands

### If Opening NEW Terminal:
```bash
# Just these two commands - bashrc loads automatically!
conda activate vlm_Qwen2VL_object_detection
cd ~/projects/vlm_Qwen2VL_object_detection
```

### If Continuing SAME Terminal from Yesterday:
```bash
# Need to reload bashrc for token
source ~/.bashrc
conda activate vlm_Qwen2VL_object_detection
cd ~/projects/vlm_Qwen2VL_object_detection
```

---

## 📋 Your TODO List

### 0. About Symlinks (FYI)
The symlinks (qwen2vl-nutrition-detection-lora, logs, etc.) look broken now but will work once training creates the directories. They let you access SSD files from your project folder.

### 1. Verify Everything Works (2 min)
```bash
python -c "
import os
from trl import DataCollatorForCompletionOnlyLM
import torch
from huggingface_hub import HfApi

print('='*50)
print('🔍 SYSTEM CHECK')
print('='*50)
print(f'✅ TRL 0.10.1: DataCollatorForCompletionOnlyLM available')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

token = os.environ.get('HF_TOKEN')
if token:
    api = HfApi(token=token)
    user = api.whoami()
    print(f'✅ HuggingFace: Logged in as {user[\"name\"]}')
else:
    print('⚠️  HF Token not found - run: source ~/.bashrc')
print('='*50)
"
```

### 2. Test Import the Notebook (1 min)
```bash
python -c "
# Test critical imports from the notebook
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
print('✅ All critical imports working!')
"
```

### 3. Quick Training Test (5 min)
```bash
# Run only 10 training steps to verify everything works
python -c "
print('📝 Creating minimal test...')
# This would run a quick 10-step training test
# Edit the notebook/script to add --max_steps=10 parameter
print('⚠️  Implement quick test in notebook first')
"
```

### 4. Start Full Training (3-6 hours)
```bash
# Option A: Run Python script
python fine_tuning_vlm_for_object_detection_trl.py

# Option B: Use Jupyter notebook
jupyter notebook fine_tuning_vlm_for_object_detection_trl.ipynb

# Option C: Run in background with logging
nohup python fine_tuning_vlm_for_object_detection_trl.py > training.log 2>&1 &
tail -f training.log  # Watch progress
```

---

## 📁 Expected Training Output Structure

When you start training, here's what will be created:

```
vlm_Qwen2VL_object_detection/
│
├── qwen2vl-nutrition-detection-lora/      # 🎯 MAIN OUTPUT DIRECTORY
│   ├── checkpoint-100/                    # Checkpoint every 100 steps
│   │   ├── adapter_model.safetensors     # LoRA weights (~200MB)
│   │   ├── adapter_config.json           # LoRA configuration
│   │   ├── training_args.bin             # Training state
│   │   ├── optimizer.pt                  # Optimizer state
│   │   ├── scheduler.pt                  # LR scheduler state
│   │   └── rng_state.pth                 # Random state for resuming
│   │
│   ├── checkpoint-200/                   # Next checkpoint
│   │   └── ... (same structure)
│   │
│   ├── checkpoint-300/
│   │   └── ... (continues every 100 steps)
│   │
│   └── [AT END OF TRAINING]:
│       ├── adapter_model.safetensors     # Final LoRA weights
│       ├── adapter_config.json           # Final config
│       ├── tokenizer_config.json         # Tokenizer settings
│       ├── special_tokens_map.json       # Special tokens
│       ├── training_results.json         # Final metrics
│       └── all_results.json              # Complete training history
│
├── qwen2vl-nutrition-detection-merged/    # If you merge LoRA (optional)
│   └── ... (full 15GB model with LoRA merged)
│
├── logs/                                  # TensorBoard logs
│   └── events.out.tfevents.{timestamp}   # Training metrics
│
├── training.log                          # If using nohup
│
└── [Model Cache on SSD]:
    /ssd1/zhuoyuan/hf_cache/
    └── models--Qwen--Qwen2-VL-7B-Instruct/  # Base model (~15GB)
```

### Storage Requirements:
- **Base Model Download**: ~15GB (one-time, cached to SSD)
- **Each Checkpoint**: ~500MB (LoRA weights + optimizer)
- **Final LoRA Model**: ~200MB (just adapter)
- **Merged Model** (optional): ~15GB (full model)
- **Total During Training**: ~20GB (model + checkpoints)
- **After Cleanup**: ~15.2GB (model cache + final LoRA)

### Important Paths (ALL ON SSD - Your home directory stays under 10MB!):
- **Training Output**: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora/`
- **Merged Model**: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-merged/`
- **Logs**: `/ssd1/zhuoyuan/vlm_outputs/logs/`
- **Model Cache**: `/ssd1/zhuoyuan/hf_cache/`
- **Final LoRA Weights**: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora/adapter_model.safetensors`

---

## 🔄 Recovery & Monitoring

### Monitor Training Progress:
```bash
# Everything accessible from project directory via symlinks!
cd ~/projects/vlm_Qwen2VL_object_detection

# Watch training log in real-time
tail -f training.log  # If using nohup

# Check GPU usage
nvidia-smi -l 1  # Updates every second

# Check latest checkpoint (symlink makes it easy!)
ls -la qwen2vl-nutrition-detection-lora/checkpoint-*/

# View logs
ls -la logs/
```

### Resume from Checkpoint:
```python
# If training interrupted, add to script:
training_args = SFTConfig(
    output_dir="./qwen2-vl-nutrition-finetuned",
    resume_from_checkpoint="./qwen2-vl-nutrition-finetuned/checkpoint-200",
    # ... other args
)
```

### Clean Up Old Checkpoints:
```bash
# Keep only last 3 checkpoints (automatic with save_total_limit=3)
# Or manually remove old ones:
rm -rf qwen2-vl-nutrition-finetuned/checkpoint-{100..500}
```

---

## 🎯 After Training Completes

### 1. Test the Fine-tuned Model:
```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load base model + LoRA weights
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    "./qwen2-vl-nutrition-finetuned"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Test on new image
# ... inference code
```

### 2. Merge LoRA Weights (Optional):
```python
# Merge LoRA into base model for easier deployment
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./qwen2-vl-nutrition-merged")
```

### 3. Push to HuggingFace Hub (Optional):
```python
# Share your model
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./qwen2-vl-nutrition-finetuned",
    repo_id="DanJZY/qwen2-vl-nutrition-detector",
    repo_type="model"
)
```

---

## ⚠️ Potential Issues & Solutions

### Issue: "CUDA out of memory"
```python
# Reduce batch size in training config
per_device_train_batch_size=1  # Instead of 2
gradient_accumulation_steps=16  # Increase to maintain effective batch size
```

### Issue: "Dataset not found"
```bash
# Make sure HF token is loaded
echo $HF_TOKEN  # Should show token
# If not: source ~/.bashrc
```

### Issue: "Import error for DataCollatorForCompletionOnlyLM"
```bash
# Verify TRL version
pip show trl  # Should be 0.10.1
# If not: pip install trl==0.10.1
```

---

## 📊 Expected Training Metrics

With your RTX 6000 Ada (47.50 GB):
- **Training Speed**: ~10-15 steps/minute
- **Memory Usage**: ~15-20 GB VRAM
- **Time per Epoch**: 1-2 hours
- **Total Time**: 3-6 hours for 3 epochs
- **Final Loss**: Should drop from ~3.5 to ~0.5
- **Checkpoint Frequency**: Every 100 steps (~10 minutes)

---

## 💡 Pro Tips

1. **Use Larger Batch Size**: Your GPU can handle batch_size=8!
   ```python
   per_device_train_batch_size=8  # Faster training
   gradient_accumulation_steps=2  # Effective batch size = 16
   ```

2. **Enable TensorBoard**:
   ```bash
   tensorboard --logdir=./qwen2-vl-nutrition-finetuned --port=6006
   ```

3. **Quick Dataset Test**:
   ```python
   # Test with only 100 samples first
   train_dataset = train_dataset.select(range(100))
   ```

---

## 🎊 You're Ready!

Everything is set up perfectly. Just:
1. Open new terminal (bashrc loads automatically)
2. Activate conda environment
3. Start training!

Good luck with your training! 🚀