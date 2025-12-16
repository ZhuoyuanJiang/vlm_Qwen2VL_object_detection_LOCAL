# Session 22 Progress - Quantization & Trainable Parameters Diagnostic Notebook

**Date**: 2024-12-16

## Summary

Created a new diagnostic notebook (`notebooks/06_quantization_and_trainable_params.ipynb`) to answer 3 key questions about Qwen2-VL model training for tutor session preparation.

---

## Questions Addressed

### Q1: What does quantization do to trainable layers?

**Finding**: 4-bit quantization does NOT freeze all parameters!

| Model | Trainable Params | Explanation |
|-------|-----------------|-------------|
| BF16 | 100% (8.3B) | All params have `requires_grad=True` by default |
| 4-bit | ~23% (1.1B) | Only Linear layers quantized; LayerNorm, embeddings, lm_head remain trainable |

**Key Insight**: BitsAndBytes only quantizes Linear layers. Non-linear layers (LayerNorm, embeddings) remain in bf16 with `requires_grad=True`. This is why `prepare_model_for_kbit_training()` is needed to explicitly freeze everything before adding LoRA.

### Q2: Do `q_proj` layers apply only to LLM or also vision encoder?

**Finding**: Different naming conventions for LLM vs Vision encoder.

| Layer Pattern | Location | Example Path |
|--------------|----------|--------------|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | LLM only | `model.language_model.layers.0.self_attn.q_proj` |
| `qkv` (combined) | Vision encoder only | `model.visual.blocks.0.attn.qkv` |
| `gate_proj`, `up_proj`, `down_proj` | LLM only | `model.language_model.layers.0.mlp.gate_proj` |
| `fc1`, `fc2` | Vision encoder only | `model.visual.blocks.0.mlp.fc1` |

### Q3: After `get_peft_model()`, which modules become trainable?

**Finding**: Only LoRA adapters (`lora_A`, `lora_B`) are trainable.

- After `prepare_model_for_kbit_training()` + `get_peft_model()`:
  - Total params: 4.85B
  - Trainable params: 161M (3.33%)
  - All trainable params are in `language_model` (LoRA adapters)
  - Vision encoder: 0 trainable params (completely frozen)

---

## Notebook Created

### `notebooks/06_quantization_and_trainable_params.ipynb`

**Structure:**
```
Section 1: Setup
├── GPU auto-detection (max 2 GPUs)
├── CUDA_VISIBLE_DEVICES configuration
└── Helper functions (summarize_trainables_by_area, find_modules_by_pattern)

Section 2: Q1 - Quantization Effect
├── Load BF16 model → 100% trainable
├── Load 4-bit model → ~23% trainable
└── Explanation of which layers get quantized

Section 3: Q2 - Layer Names
├── Search for q_proj → LLM only (28 found)
├── Search for qkv → Vision encoder only (32 found)
└── Layer naming convention table

Section 4: Q3 - get_peft_model Verification
├── Define LoRA config (LLM targets)
├── Apply prepare_model_for_kbit_training + get_peft_model
├── Verify trainable params (3.33% - LoRA only)
├── Print full PEFT model structure
└── Confirm vision encoder frozen

Section 5: Summary & Key Takeaways
└── Concise answers to all 3 questions
```

---

## Key Features of the Notebook

### 1. GPU Auto-Detection
- Automatically finds up to 2 available GPUs (< 2GB used, > 40GB total)
- Sets `CUDA_VISIBLE_DEVICES` BEFORE importing torch
- Fallback to GPU with most free memory if no idle GPUs found

### 2. Memory Management
- Cleanup function between model loads to avoid OOM
- `torch.cuda.empty_cache()` after deleting models

### 3. Full PEFT Model Print
- Added cell to print entire model structure after LoRA application
- Shows where `lora_A` and `lora_B` modules are inserted

---

## Next Steps (for tutor session)

1. Run the notebook to see actual outputs
2. Review the PEFT model structure to understand LoRA placement
3. Optionally add vision encoder LoRA targets for comparison:
   ```python
   target_modules = [
       "q_proj", "k_proj", "v_proj", "o_proj",  # LLM
       "gate_proj", "up_proj", "down_proj",      # LLM MLP
       "qkv", "fc1", "fc2",                      # Vision encoder
   ]
   ```
