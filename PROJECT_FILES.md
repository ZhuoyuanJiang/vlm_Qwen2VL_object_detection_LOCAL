# 📂 Project Files Overview

## Core Training Files
| File | Purpose | When to Use |
|------|---------|-------------|
| `fine_tuning_vlm_for_object_detection_trl.py` | Main training script | Run this to start training |
| `fine_tuning_vlm_for_object_detection_trl.ipynb` | Jupyter notebook version | For interactive development |
| `requirements.txt` | Python dependencies | Already installed |

## Documentation Files
| File | Purpose | Read When |
|------|---------|-----------|
| `README.md` | Project overview & quick start | Start here |
| `DO_AFTER_SLEEP.md` | Step-by-step guide for training | **Read this first after waking up** |
| `QUICK_REFERENCE.md` | Commands cheat sheet | Keep open while training |
| `PROJECT_ANALYSIS.md` | Detailed hyperparameter rationale | Understand design choices |
| `DETAILED_SESSION_LOG.md` | Complete setup history | If troubleshooting |
| `WORKFLOW_GUIDE.md` | Original workflow documentation | Additional context |
| `PROJECT_FILES.md` | This file - explains all files | Now! |

## Test & Development Files
| File/Directory | Purpose | Status |
|----------------|---------|--------|
| `test_notebook_sections.py` | Tests for notebook sections | Completed |
| `test_outputs/` | Test results and reports | Contains test logs |
| `.gitignore` | Git ignore rules | Configured |

## Symlinks (Created for Easy Access)
| Symlink Name | Points To | Purpose |
|--------------|-----------|---------|
| `qwen2vl-nutrition-detection-lora` | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora/` | Your trained LoRA model |
| `qwen2vl-nutrition-detection-merged` | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-merged/` | Merged full model (optional) |
| `logs` | `/ssd1/zhuoyuan/vlm_outputs/logs/` | Training logs |
| `model_cache` | `/ssd1/zhuoyuan/hf_cache/` | Downloaded base models |

**Note**: These appear as broken links now but will work once training creates the directories on SSD.

## Output Directories (Created During Training on SSD)
| Actual Location on SSD | Contents | Size |
|------------------------|----------|------|
| `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora/` | **Your trained model** | ~200MB final |
| `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-lora/checkpoint-*/` | Training checkpoints | ~500MB each |
| `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-merged/` | Merged model (optional) | ~15GB |
| `/ssd1/zhuoyuan/vlm_outputs/logs/` | TensorBoard logs | ~50MB |
| `training.log` | Training output (if using nohup) | Text file in project dir |

## Hidden/System Files
| File/Directory | Purpose | Note |
|----------------|---------|------|
| `.git/` | Git repository data | Don't modify |
| `.claude/` | Claude Code metadata | Don't modify |
| `*.Zone.Identifier` | Windows WSL metadata | Can ignore |

## What You Need to Focus On

### For Training:
1. **Read**: `DO_AFTER_SLEEP.md`
2. **Run**: `fine_tuning_vlm_for_object_detection_trl.py`
3. **Monitor**: `training.log` and `qwen2vl-nutrition-detection-lora/`

### For Understanding:
1. **Architecture**: `PROJECT_ANALYSIS.md`
2. **Quick Commands**: `QUICK_REFERENCE.md`

### Your Trained Model Will Be:
- **LoRA Weights**: `qwen2vl-nutrition-detection-lora/adapter_model.safetensors`
- **Config**: `qwen2vl-nutrition-detection-lora/adapter_config.json`

## File Sizes
- Python script: ~60KB
- Notebook: ~80KB
- Documentation: ~200KB total
- Trained LoRA model: ~200MB
- Base model cache: ~15GB (on SSD)