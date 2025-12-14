# Plan: Diagnostic Verification & 4 Training Recipes

**Date:** 2025-12-14
**Goal:** Add diagnostic tools and implement 4 training recipes

---

## Organization Decision

| Content | Location |
|---------|----------|
| **Model architecture inspection** (module names, Linear layers) | `notebooks/02_model_understanding.ipynb` |
| **Quantization inspection** (bf16 vs 4-bit trainable params) | `fine_tuning_vlm_for_object_detection_trl.py` (main file) |
| **LoRA inspection** (verify targets LLM-only) | `fine_tuning_vlm_for_object_detection_trl.py` (main file) |
| **4 Training Recipes** | `fine_tuning_vlm_for_object_detection_trl.py` (end of notebook, after v2 training) |

**Rationale:** Main file is already long. Architecture inspection is educational/one-time, fits well in model understanding notebook. Quantization/LoRA inspection is training-related, stays in main file. Recipes use AssistantOnlyCollator and go at the end.

---

## Part 1: Model Architecture Inspection (02_model_understanding.ipynb)

### New Section: "Model Architecture - Layer Names"

Add after existing "Model Loading and Configuration" section (~cell 5).

**Code to add:**

```python
# ============================================================
# MODEL ARCHITECTURE - LAYER NAMES FOR LORA TARGETING
# ============================================================
# This section helps you understand which layers exist in Qwen2-VL
# so you can configure LoRA target_modules correctly.

import torch.nn as nn

def list_linear_modules(model, prefix_filter=None, max_print=50):
    """
    List all nn.Linear modules in the model.

    Args:
        model: The model to inspect
        prefix_filter: If provided, only show modules starting with this prefix
        max_print: Maximum number of modules to print

    Returns:
        List of module names
    """
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if prefix_filter is None or name.startswith(prefix_filter):
                linear_modules.append(name)

    print(f"\n{'='*60}")
    print(f"Linear modules" + (f" under '{prefix_filter}'" if prefix_filter else ""))
    print(f"{'='*60}")
    print(f"Total found: {len(linear_modules)}")

    for i, name in enumerate(linear_modules[:max_print]):
        print(f"  {i+1:3d}. {name}")

    if len(linear_modules) > max_print:
        print(f"  ... ({len(linear_modules) - max_print} more)")

    return linear_modules


def find_modules_by_pattern(model, pattern):
    """
    Find all modules containing a specific pattern in their name.

    Args:
        model: The model to inspect
        pattern: Substring to search for (e.g., "q_proj", "qkv")

    Returns:
        List of matching module names
    """
    matches = []
    for name, _ in model.named_modules():
        if pattern in name:
            matches.append(name)

    print(f"\n{'='*60}")
    print(f"Modules containing '{pattern}'")
    print(f"{'='*60}")
    print(f"Total found: {len(matches)}")

    for name in matches[:30]:
        print(f"  {name}")

    if len(matches) > 30:
        print(f"  ... ({len(matches) - 30} more)")

    return matches


# ============================================================
# INSPECT VISION ENCODER LAYERS
# ============================================================
print("\n" + "="*80)
print("VISION ENCODER ARCHITECTURE")
print("="*80)

vision_linears = list_linear_modules(model, prefix_filter="visual")

# ============================================================
# INSPECT LLM DECODER LAYERS
# ============================================================
print("\n" + "="*80)
print("LLM DECODER ARCHITECTURE")
print("="*80)

llm_linears = list_linear_modules(model, prefix_filter="model.layers")

# ============================================================
# VERIFY LORA TARGET MODULE LOCATIONS
# ============================================================
print("\n" + "="*80)
print("LORA TARGET MODULE VERIFICATION")
print("="*80)

# Check where q_proj exists (should be LLM only)
print("\n--- Where does 'q_proj' exist? ---")
q_proj_modules = find_modules_by_pattern(model, "q_proj")

# Check where qkv exists (should be vision encoder only)
print("\n--- Where does 'qkv' exist? ---")
qkv_modules = find_modules_by_pattern(model, "qkv")

# Check for potential conflicts
print("\n--- Where does 'proj' exist? (potential conflict check) ---")
proj_modules = find_modules_by_pattern(model, "proj")

# Summary
print("\n" + "="*80)
print("SUMMARY: Module Name Conventions")
print("="*80)
print("""
VISION ENCODER (visual.blocks[i]):
  - Attention: qkv (combined Q,K,V), proj (output)
  - MLP: fc1, fc2

LLM DECODER (model.layers[i]):
  - Attention: q_proj, k_proj, v_proj, o_proj
  - MLP: gate_proj, up_proj, down_proj (SwiGLU)

CURRENT LORA TARGETS: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  → These target LLM ONLY (not vision encoder)

TO TARGET VISION ENCODER: qkv, proj, fc1, fc2
  → WARNING: 'proj' might also match 'o_proj' - verify with inspection!
""")
```

---

## Part 2: Quantization & LoRA Inspection (main file)

### Location: After model loading, before LoRA config (~line 1445)

**New section title:** `# === DIAGNOSTIC: BF16 vs 4-bit Trainable Parameters ===`

```python
# ============================================================
# DIAGNOSTIC: BF16 vs 4-bit Trainable Parameters
# ============================================================
# This section compares what's trainable with and without quantization.
# Run this once to understand the effect of 4-bit quantization.
# You can skip this section in production training runs.

def summarize_trainables_by_area(model, title="Model"):
    """
    Summarize trainable parameters grouped by model area.
    """
    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)

    pct = 100 * trainable_params / max(total_params, 1)

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({pct:.4f}%)")

    # Count by area
    areas = {"visual": 0, "model.layers": 0, "lm_head": 0, "other": 0}
    for name in trainable_names:
        if name.startswith("visual") or ".visual." in name:
            areas["visual"] += 1
        elif "model.layers" in name or name.startswith("model."):
            areas["model.layers"] += 1
        elif "lm_head" in name:
            areas["lm_head"] += 1
        else:
            areas["other"] += 1

    print(f"\nTrainable parameter counts by area:")
    for area, count in areas.items():
        print(f"  {area:20s}: {count}")

    return trainable_names

# --- Compare BF16 vs 4-bit (uncomment to run) ---
# WARNING: This loads the model twice, uses significant GPU memory!
# Only run this section once for diagnostic purposes.

"""
# BF16 load (no quantization)
model_bf16 = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
summarize_trainables_by_area(model_bf16, "BF16 Base Model (no quantization)")
del model_bf16
torch.cuda.empty_cache()

# 4-bit load
model_4bit = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
summarize_trainables_by_area(model_4bit, "4-bit Base Model (before LoRA)")
del model_4bit
torch.cuda.empty_cache()
"""

# ============================================================
# DIAGNOSTIC: LoRA Target Verification
# ============================================================
# This temporarily applies get_peft_model to verify LoRA targets.
# FOR INSPECTION ONLY - do NOT use this model for training!

"""
from peft import LoraConfig, get_peft_model

# Fresh model load for inspection
model_inspect = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    attn_implementation="sdpa",
    trust_remote_code=True,
)

# Apply current LoRA config (LLM targets)
llm_lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

peft_model = get_peft_model(model_inspect, llm_lora_config)
trainable_names = summarize_trainables_by_area(peft_model, "4-bit + LLM LoRA (get_peft_model)")

# Verify no vision params are trainable
vision_trainables = [n for n in trainable_names if "visual" in n]
print(f"\nVision encoder trainable params: {len(vision_trainables)}")
if len(vision_trainables) == 0:
    print("✅ CONFIRMED: LoRA targets LLM only, not vision encoder")
else:
    print("⚠️ WARNING: Some vision params are trainable!")
    for n in vision_trainables[:10]:
        print(f"  {n}")

del peft_model, model_inspect
torch.cuda.empty_cache()
"""
```

---

## Part 3: Training Recipes (end of main file)

### Location: After v2 training section (~line 2950), before evaluation

**All recipes use AssistantOnlyCollator (the better collator we discovered).**

```python
# ============================================================
# TRAINING RECIPES FRAMEWORK
# ============================================================
# Four recipes to compare different training approaches:
# - Recipe 1 (r1): LLM-only QLoRA (current baseline)
# - Recipe 2 (r2): Vision-only full fine-tuning
# - Recipe 3 (r3): Two-stage (vision first, then LLM)
# - Recipe 4 (r4): Joint (vision LoRA + LLM LoRA)

from dataclasses import dataclass
from typing import Optional, List

# --- Target Module Constants ---
LLM_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # MLP (SwiGLU)
]

VISION_LORA_TARGETS = [
    "qkv",   # Combined Q,K,V attention
    "fc1",   # MLP first layer
    "fc2",   # MLP second layer
    # NOTE: Omitting "proj" to avoid matching "o_proj" in LLM
]


@dataclass
class TrainingRecipe:
    """Configuration for a training recipe."""
    name: str
    quantized: bool              # True = 4-bit, False = bf16
    train_vision: str            # "none" | "full" | "lora"
    train_llm: str               # "none" | "full" | "lora"
    lora_target_modules: Optional[List[str]]
    learning_rate: float
    description: str


RECIPES = {
    "r1-llm-only": TrainingRecipe(
        name="r1-llm-only",
        quantized=True,
        train_vision="none",
        train_llm="lora",
        lora_target_modules=LLM_LORA_TARGETS,
        learning_rate=1e-4,
        description="Baseline: LLM-only QLoRA (4-bit), vision frozen",
    ),
    "r2-vision-only": TrainingRecipe(
        name="r2-vision-only",
        quantized=False,  # bf16 for full fine-tuning
        train_vision="full",
        train_llm="none",
        lora_target_modules=None,
        learning_rate=5e-5,
        description="Vision encoder full fine-tuning, LLM frozen",
    ),
    "r3-two-stage": TrainingRecipe(
        name="r3-two-stage",
        quantized=True,  # Stage 2 uses 4-bit
        train_vision="none",  # Stage 1 handled separately
        train_llm="lora",
        lora_target_modules=LLM_LORA_TARGETS,
        learning_rate=1e-4,
        description="Two-stage: Vision first (r2) → LLM (r1) from checkpoint",
    ),
    "r4-joint": TrainingRecipe(
        name="r4-joint",
        quantized=True,
        train_vision="lora",
        train_llm="lora",
        lora_target_modules=LLM_LORA_TARGETS + VISION_LORA_TARGETS,
        learning_rate=1e-4,
        description="Joint: Vision LoRA + LLM LoRA together",
    ),
}


def load_model_for_recipe(model_id: str, recipe: TrainingRecipe, checkpoint_path: str = None):
    """Load model configured for a specific recipe."""
    bnb_config = None
    if recipe.quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    load_path = checkpoint_path if checkpoint_path else model_id

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        load_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    processor = Qwen2VLProcessor.from_pretrained(
        model_id,  # Always use original for processor
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    return model, processor


def apply_freeze_policy(model, recipe: TrainingRecipe):
    """Apply freeze/unfreeze based on recipe configuration."""
    # Start with everything frozen
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on recipe
    if recipe.train_vision == "full":
        for name, param in model.named_parameters():
            if name.startswith("visual") or ".visual." in name:
                param.requires_grad = True

    if recipe.train_llm == "full":
        for name, param in model.named_parameters():
            if "model.layers" in name or "lm_head" in name:
                param.requires_grad = True

    # For LoRA cases, PEFT will handle unfreezing adapter weights


def build_lora_config_for_recipe(recipe: TrainingRecipe) -> Optional[LoraConfig]:
    """Build LoRA config for a recipe, or None if not using LoRA."""
    if recipe.train_llm != "lora" and recipe.train_vision != "lora":
        return None

    return LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recipe.lora_target_modules,
    )


def run_recipe(recipe_key: str, model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
               checkpoint_path: str = None, num_epochs: int = 3):
    """
    Run a complete training recipe.

    Args:
        recipe_key: Key from RECIPES dict (e.g., "r1-llm-only")
        model_id: HuggingFace model ID
        checkpoint_path: Optional path to load from (for two-stage)
        num_epochs: Number of training epochs

    Returns:
        Output directory path
    """
    recipe = RECIPES[recipe_key]

    print("\n" + "="*80)
    print(f"RUNNING RECIPE: {recipe.name}")
    print(f"Description: {recipe.description}")
    print("="*80)

    # Load model
    model, processor = load_model_for_recipe(model_id, recipe, checkpoint_path)

    # Apply freeze policy
    apply_freeze_policy(model, recipe)

    # Build LoRA config
    lora_config = build_lora_config_for_recipe(recipe)

    # Create training config
    training_args = create_training_config(
        variant=recipe.name,
        learning_rate=recipe.learning_rate,
        num_train_epochs=num_epochs,
    )

    # Use AssistantOnlyCollator for all recipes
    collator = AssistantOnlyCollator(processor, model)
    collator.debug = False

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_formatted,
        eval_dataset=eval_dataset_formatted,
        data_collator=collator,
        processing_class=processor,
        peft_config=lora_config,
    )

    print("\nTrainable parameters after trainer creation:")
    trainer.model.print_trainable_parameters()

    # Train
    trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    print(f"\n✅ Recipe {recipe.name} complete! Saved to: {training_args.output_dir}")

    return training_args.output_dir


def run_two_stage_recipe(model_id: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """
    Run the two-stage recipe (r3):
    Stage 1: Vision-only training (r2)
    Stage 2: LLM-only training from Stage 1 checkpoint (r1)
    """
    print("\n" + "="*80)
    print("TWO-STAGE RECIPE (r3)")
    print("Stage 1: Vision encoder training")
    print("Stage 2: LLM training from Stage 1 checkpoint")
    print("="*80)

    # Stage 1: Vision-only
    stage1_dir = run_recipe("r2-vision-only", model_id)

    # Stage 2: LLM-only from Stage 1 checkpoint
    stage2_dir = run_recipe("r1-llm-only", model_id, checkpoint_path=stage1_dir)

    return stage2_dir


# --- Example Usage ---
# Uncomment the recipe you want to run:

# Recipe 1: LLM-only (baseline)
# output_dir = run_recipe("r1-llm-only")

# Recipe 2: Vision-only
# output_dir = run_recipe("r2-vision-only")

# Recipe 3: Two-stage
# output_dir = run_two_stage_recipe()

# Recipe 4: Joint (run after verifying VISION_LORA_TARGETS don't conflict)
# output_dir = run_recipe("r4-joint")
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `notebooks/02_model_understanding.ipynb` | Add "Model Architecture - Layer Names" section |
| `fine_tuning_vlm_for_object_detection_trl.py` | Add diagnostics (~line 1445) + recipes (end of file) |

---

## Implementation Order

1. **Part 1:** Add architecture inspection to `02_model_understanding.ipynb`
2. **Part 2:** Add quantization/LoRA diagnostics to main file
3. **Part 3:** Add recipe framework to end of main file
4. **Verify:** Run diagnostics to confirm module names before running recipes
5. **Train:** Run recipes one at a time, compare results
