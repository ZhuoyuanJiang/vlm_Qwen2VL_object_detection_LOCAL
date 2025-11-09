# Handover to Next Session - November 9, 2025

## 🎯 Where We Left Off

We completed the **first phase of refactoring** to enable easier debugging of training issues.

### ✅ What's Done:
1. Created refactoring plan (`refactor_documentation/refactor_plan_20251109.md`)
2. Extracted dataset EDA to notebook (`notebooks/01_dataset_exploration.ipynb`)
3. Created `src/data/dataset.py` - Dataset preparation functions (runs ONCE)
4. Created `src/data/collators.py` - Batch collation helpers (runs EVERY batch)
5. Created test suite (`tests/test_data_format_before_chat_template.py`)
6. Created golden output test skeleton (`tests/test_golden_output.py`)
7. Documented everything (`refactor_documentation/`)

---

## 🚀 What to Do Next Session

### **Phase 1: Extract Working Modules (supposed to work parts)**

**Goal**: Extract modules that are already working correctly (inference, visualization, GPU utils)

**Why First?**: User confirmed these parts work correctly. Focus testing on collators where bugs likely are.

#### Step 1: Extract Model Inference (15 min)
**Create**: `src/models/inference.py`

**Extract from main file**:
- `run_qwen2vl_inference()` (lines ~952-1108)
- `parse_qwen_bbox_output()` (lines ~1109-1164)

**Lines to extract**: ~952-1164 (212 lines)

**Documentation template**:
```python
"""
Model inference functions for Qwen2-VL.

WHEN USED: For testing model before/after training
RUNS: On-demand for inference

Functions:
1. run_qwen2vl_inference(image, prompt, model_id)
   - WHEN: Test model on single image
   - INPUT: Image path/PIL image + prompt
   - OUTPUT: Model prediction text

2. parse_qwen_bbox_output(text)
   - WHEN: Parse model output to get bounding boxes
   - INPUT: Model prediction text with special tokens
   - OUTPUT: List of (category, bbox) tuples
"""
```

#### Step 2: Extract Visualization (10 min)
**Create**: `src/utils/visualization.py`

**Extract from main file**:
- `visualize_bbox_on_image()` (lines ~1165-1200)

**Lines to extract**: ~1165-1200 (35 lines)

#### Step 3: Extract GPU Utils (10 min)
**Create**: `src/utils/gpu.py`

**Extract from main file**:
- `find_available_gpus()` and GPU setup code (lines ~36-179)

**Lines to extract**: ~36-179 (143 lines)

---

### **Phase 2: Complete Golden Output Tests (20 min)**

**File**: `tests/test_golden_output.py` (skeleton already created!)

**Steps**:
1. Activate conda environment:
   ```bash
   conda activate vlm_Qwen2VL_object_detection
   ```

2. Load one datapoint and get actual output:
   ```python
   from datasets import load_dataset
   from src.data.dataset import convert_to_conversation_format

   ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
   sample = next(iter(ds))
   result = convert_to_conversation_format(sample)

   # Print and copy this output
   print(result)
   ```

3. Copy the output into `expected_format` in `test_golden_output.py`

4. Uncomment the assertions

5. Run test to verify format

---

### **Phase 3: Run All Tests (15 min)**

**Order**:
1. Run golden output tests first (most intuitive):
   ```bash
   python tests/test_golden_output.py
   ```

2. Run comprehensive test suite:
   ```bash
   python tests/test_data_format_before_chat_template.py
   ```

3. If tests fail → Document what's wrong
4. If tests pass → Format is correct, continue extracting

---

### **Phase 4: Extract Collator Classes (30 min) - After Tests**

**IMPORTANT**: Only do this AFTER tests pass or you understand what's wrong

**Create collator classes in**: `src/data/collators.py`

**Extract from main file**:
- `collate_fn_fixed_1` / `Qwen2VLCollatorV1` (lines ~2452-2609)
- `collate_fn_fixed_fixed1` / `Qwen2VLCollatorV2` (lines ~2602-2787)
- `collate_fn_fixed_3` / `Qwen2VLCollatorV3` (lines ~4050+)

**Keep all 3 versions** - user wants them for educational purposes (to show students the evolution)

---

## 📁 Current Project Structure

```
vlm_Qwen2VL_object_detection/
│
├── refactor_documentation/          ✅ All docs here
│   ├── README.md
│   ├── refactor_plan_20251109.md
│   ├── PROGRESS_20251109.md
│   └── HANDOVER_TO_NEXT_SESSION.md  ← YOU ARE HERE
│
├── src/                             ✅ Extracted modules
│   ├── data/
│   │   ├── dataset.py               ✅ Dataset preparation (runs ONCE)
│   │   └── collators.py             ✅ Batch collation helpers (runs EVERY batch)
│   ├── models/                      📝 TODO: Add inference.py
│   ├── training/                    📝 TODO: Add callbacks.py later
│   └── utils/                       📝 TODO: Add gpu.py, visualization.py
│
├── tests/
│   ├── test_data_format_before_chat_template.py  ✅ Comprehensive tests
│   └── test_golden_output.py                     📝 TODO: Complete
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb              ✅ Dataset EDA
│
└── fine_tuning_vlm_for_object_detection_trl.py  (main file - still monolithic)
```

---

## 🔍 Key Context to Remember

### User's Goal:
Fix training bugs (validation loss doesn't decrease, trained model worse than original)

### User's Insight:
> "Everything before the training part should have no much problem... like run model inference before fine-tuning. Those should be correct already."

This means:
- ✅ Inference code is likely correct
- ✅ Visualization is likely correct
- ✅ GPU management is likely correct
- ❓ **Bugs are likely in collators or training callbacks**

### Strategy:
1. Extract "working" parts first → modular, testable
2. Focus testing on collators → find the bug
3. Fix bug in isolated module → easier than monolithic file

---

## 💡 Quick Commands for Next Session

```bash
# Navigate to project
cd /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection

# Activate environment
conda activate vlm_Qwen2VL_object_detection

# Read documentation
cat refactor_documentation/HANDOVER_TO_NEXT_SESSION.md
cat refactor_documentation/PROGRESS_20251109.md

# Check current structure
ls -la src/data/
ls -la tests/

# Start with extraction (see Phase 1 above)
# Then complete golden output tests (see Phase 2)
# Then run all tests (see Phase 3)
```

---

## 📝 Files to Create Next Session

**In order**:
1. `src/models/inference.py` (inference functions)
2. `src/utils/visualization.py` (bbox visualization)
3. `src/utils/gpu.py` (GPU management)
4. Complete `tests/test_golden_output.py` (fill in expected formats)
5. Later: Extract collator classes after tests pass

---

## ⚠️ Important Notes

### Don't Modify Extracted Code!
- Extract = COPY EXACT code, NO modifications
- If bugs exist, find them through tests first
- Then fix knowing exactly what's wrong

### Keep Educational Value
- Keep all 3 collator versions (V1, V2, V3)
- Show students the evolution of fixing bugs
- Document why each version is different

### Documentation Standards
- Add QUICK REFERENCE at top of each module
- Add QUICK REFERENCE in each function docstring
- Show INPUT/OUTPUT format examples
- Explain WHEN code runs (once vs every batch)

---

## 🎯 Expected Timeline

**Next session should take ~1.5-2 hours**:
- 35 min: Extract working modules (inference, visualization, GPU)
- 20 min: Complete golden output tests
- 15 min: Run all tests
- 30 min: Extract collator classes (if tests pass)
- 10 min: Update main file to import from modules
- 10 min: Document findings

---

## 🤔 Decision Points for Next Session

### If Tests Pass:
- ✅ Format is correct
- Continue extracting collator classes
- Look for bugs elsewhere (training loop, callbacks)

### If Tests Fail:
- ❌ Found the format bug!
- Document exactly what's wrong
- Fix in isolated module
- Re-run tests to verify
- Then continue extraction

---

## 📚 Key Files to Reference

- **Full plan**: `refactor_documentation/refactor_plan_20251109.md`
- **Today's progress**: `refactor_documentation/PROGRESS_20251109.md`
- **Extracted modules**: `src/data/dataset.py`, `src/data/collators.py`
- **Tests**: `tests/test_data_format_before_chat_template.py`
- **Golden test skeleton**: `tests/test_golden_output.py`


