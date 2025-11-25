Extract collators

Extract all three collators to src/data/collators.py:

1. collate_fn_fixed_1 (lines 2168-2313) - Basic version, masks pad + vision tokens
2. collate_fn_fixed_fixed1 (lines 2318-2478) - With Flash Attention patch (make optional)
3. collate_fn_fixed_3 (lines 3766-3927) - Assistant-only training

Keep original names for easy mapping back to main file.

---
Extraction Details

1. collate_fn_fixed_1 (basic)

- Copy exactly from lines 2168-2313
- No modifications needed

2. collate_fn_fixed_fixed1 (with optional Flash patch)

- Copy from lines 2318-2478
- Change __init__ to accept use_flash_patch=False parameter
- Only call _patch_flash_attention() if use_flash_patch=True

3. collate_fn_fixed_3 (assistant-only)

- Copy exactly from lines 3766-3927
- No modifications needed


---
File Modified

`src/data/collators.py`