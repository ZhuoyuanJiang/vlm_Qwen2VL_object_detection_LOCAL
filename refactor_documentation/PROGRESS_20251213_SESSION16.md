# Session 16 Progress - Fix and Debug Evaluation Notebook

**Date:** 2025-12-13
**Duration:** ~2 hours
**Outcome:** Fixed multiple bugs in evaluation notebook, discovered checkpoint server issue, backed up correct checkpoints

---

## Executive Summary

This session focused on debugging the evaluation notebook (`04_evaluation_analysis.ipynb`) which was giving incorrect IoU results (~0.47 instead of ~0.78-0.81). After extensive investigation, we discovered the user was on the wrong server with old buggy checkpoints. After switching servers, we fixed several bugs and helped backup the correct checkpoints to Google Drive.

---

## Issues Found and Fixed

### 1. Dataset Split Name Error

**Problem:** `KeyError: 'validation'`

**Fix:** Changed dataset split from `'test'` to `'val'` (line 97)
```python
# BEFORE:
eval_dataset = dataset['test']

# AFTER:
eval_dataset = dataset['val']  # OpenFoodFacts uses 'val' split
```

### 2. GPU Setup Missing

**Problem:** Notebook was using all 8 GPUs instead of 1-2, waste GPU resources of the cluster

**Fix:** Added GPU setup before torch import (lines 40-42) to limit only using at most 2 GPUs.
```python
from src.utils.gpu import setup_gpus
setup_gpus(use_dual_gpu=True, max_gpus=2)
```

### 3. Multiple Detections Bug

**Problem:** `AttributeError: 'list' object has no attribute 'get'`

**Cause:** `parse_qwen_bbox_output()` can return either a dict (single detection) or list of dicts (multiple detections)

**Fix:** Added isinstance check (lines 412-416)
```python
if isinstance(parsed_bbox, list):
    pred_label = f"Predicted: {parsed_bbox[0].get('object', 'detected')}"
else:
    pred_label = f"Predicted: {parsed_bbox.get('object', 'detected')}"
```

### 4. CUDA Out of Memory for Base Model

**Problem:** Loading base model after v1 and v3 caused OOM errors

**Fix:** Added memory clearing before loading base model (lines 240-249)
```python
del model_v1
del model_v3
del processor_v1
del processor_v3
import gc
gc.collect()
torch.cuda.empty_cache()
```

## Checkpoint Backup to Google Drive

Files saved to: **Google Drive → My Drive → vlm_checkpoints/**

---

## Concepts Understanding

### 1. LoRA Merge (`merge_and_unload()`)

PeftModel stores:
- Base model (frozen weights W)
- LoRA config (r, alpha, target_modules)
- LoRA adapters (A, B matrices)

Merge formula: `W_merged = W + (A · B) × (alpha / r)`

### 2. Multiple Detections Handling

- `parse_qwen_bbox_output()` returns dict or list of dicts
- Current evaluation uses only first detection (appropriate for this dataset with 1 GT per image)
- Visualization draws all boxes, but label shows first only

### 3. Merged Model vs LoRA Model

| Aspect | LoRA Model | Merged Model |
|--------|------------|--------------|
| File size | ~600MB (adapters) | ~16GB (full model) |
| Requires PEFT | Yes | No |
| Inference speed | Slightly slower | Faster |

The merged model in this project is from v1 (all tokens), created Nov 26.

---

## Files Modified

| File | Changes |
|------|---------|
| `notebooks/04_evaluation_analysis.py` | Fixed dataset split, GPU setup, multiple detections bug, OOM fix |
| `notebooks/04_evaluation_analysis.ipynb` | Synced via jupytext |

---

## Current State of Checkpoints (Correct Server (13))

| Model | Path | Date | grad_norm | IoU |
|-------|------|------|-----------|-----|
| v1 | `/ssd1/.../qwen2vl-nutrition-detection-lora` | Nov 26 04:53 | >0 ✓ | ~0.78 |
| v3 | `/ssd1/.../qwen2vl-nutrition-detection-lora-assistantonly` | Nov 26 05:48 | >0 ✓ | ~0.81 |
| merged | `/ssd1/.../qwen2vl-nutrition-detection-merged` | Nov 26 05:00 | N/A | (v1 merged) |

---

## Next Steps

1. Clean up the codebase
2. Try Different training recipes (e.g. phased training).
