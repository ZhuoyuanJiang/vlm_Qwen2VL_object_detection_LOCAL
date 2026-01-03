# Session 26 Progress - Fix AssistantOnlyCollator EOS Token Bug

**Date**: 2026-01-03
**Session Name**: session26-fix-assistant-only-collator

## Summary

Discovered and fixed a critical training bug in `AssistantOnlyCollator` that caused the model to repeat its output indefinitely instead of stopping after generating the bounding box.

## Initial Investigation: vLLM vs Model Issue

We started this session investigating the vLLM repetition bug documented in `VLLM_REPETITION_BUG.md`. The initial hypothesis was that vLLM wasn't respecting the model's EOS tokens.

**Key Discovery**: By running inference with **transformers directly** (bypassing vLLM entirely), we observed the **same repetition behavior**:
- Model repeated detection 11 times
- Stopped only when hitting `max_tokens` (256)
- EOS token was never generated

**Conclusion**: This is **NOT a vLLM issue** - it's a **model training issue**. The model was never trained to generate the EOS token after the bounding box output.

## Bug Discovery

### The Problem
When running inference (both with vLLM and transformers), the model repeats its detection output:
```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)<|box_end|>
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)<|box_end|>
... (repeats until max_tokens)
```

### Why It Was Hidden
The evaluation code (`parse_qwen_bbox_output`) uses `re.findall()` to find all matches but then takes only the **first** bounding box. So evaluation metrics appeared correct even though the model was repeating.

### Root Cause
`AssistantOnlyCollator` in `src/data/collators.py` line 474 excluded `<|im_end|>` (EOS token) from training labels:

```python
# BUGGY (before):
response_text = text[response_start_in_text:assistant_end_pos]  # Excludes <|im_end|>!
```

The model was trained to generate:
```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x,y),(x,y)<|box_end|>
```

But **never trained to generate `<|im_end|>` (EOS token 151645)** after the bounding box. So it didn't know when to stop.

## The Fix

**File**: `src/data/collators.py`, line 474

**Change**:
```python
# FIXED (after):
response_text = text[response_start_in_text:assistant_end_pos + len(self.assistant_end_token)]
```

This includes `<|im_end|>` in the training labels so the model learns to stop after generating the bounding box.

## Verification

Before fix (21 tokens trained):
```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(14,57),(991,603)<|box_end|>
```

After fix (22 tokens trained):
```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(14,57),(991,603)<|box_end|><|im_end|>
```

## Files Changed

1. **`src/data/collators.py`** - Fixed `AssistantOnlyCollator` to include `<|im_end|>` in training labels

## Files Created

1. **`notebooks/debug_repetition_bug.py`** - Debug notebook documenting the bug discovery and verification

## Affected Models

All models trained with `train_recipe.py` (r1, r2, r3, r4) used `AssistantOnlyCollator` and have this bug. They need to be retrained to benefit from the fix.

## Current Workaround (for deployed models without retraining)

Use `stop: ["<|box_end|>"]` in vLLM API calls to stop generation after the first detection.

## Retraining Status

**Completed on vllab15**:
- Recipe: r4-joint (3 epochs)
- LoRA adapter: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint`
- Merged model: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged`

## vLLM Deployment Verification

**Date**: 2026-01-03

Successfully deployed the retrained model to vLLM on vllab8 and verified the fix works:

1. **Replaced old model** on vllab8 with new model from vllab15:
   ```bash
   rsync -avP vllab15:/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged/ \
     /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged/
   ```

2. **Tested WITHOUT the stop workaround** - model now stops naturally:
   ```
   Model Output (raw):
   <|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(273,494),(732,620)<|box_end|>
   ```

3. **Removed stop workaround** from test scripts:
   - `scripts/test_vllm_api.py` - removed `"stop": ["<|box_end|>"]`
   - `scripts/test_vllm_with_visualization.py` - removed `"stop": ["<|box_end|>"]`

**Result**: The fix is confirmed working in production. The model generates the bounding box and stops cleanly without needing the `stop` parameter.

## Next Steps

1. ~~Retrain at least one model (r4-joint recommended) to verify the fix eliminates repetition~~ Done!
2. ~~Verify new model no longer repeats~~ Done!
3. Retrain remaining recipes (r1, r2, r3) if needed

## Key Learnings

1. When training assistant-only, always include the EOS token (`<|im_end|>`) in the labels
2. Evaluation code that extracts "first match" can hide repetition bugs
3. String slicing for chat template parsing requires careful attention to boundary tokens
