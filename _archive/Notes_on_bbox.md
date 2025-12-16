# Notes on Bounding Box Formats

This document explains the different bbox formats used in this project and how each function handles the conversion.

---

## Format Summary

| Source | Format | Range | Example |
|--------|--------|-------|---------|
| **Ground Truth** (OpenFoodFacts) | `[y_min, x_min, y_max, x_max]` | [0, 1] | `[0.1, 0.2, 0.9, 0.8]` |
| **Predicted** (Qwen2-VL) | `[x_min, y_min, x_max, y_max]` | [0, 1000) | `[200, 100, 800, 900]` |

**The formats ARE different!** But each function handles the conversion appropriately.

---

## How Each Function Handles This

### 1. `parse_qwen_bbox_output()` (src/models/inference.py)

```python
'bbox': [x1, y1, x2, y2]  # Qwen outputs in [x,y,x,y] format
```

- **Input:** Raw model output string like `"nutrition table(200,100),(800,900)"`
- **Output:** `{'object': 'nutrition table', 'bbox': [200, 100, 800, 900]}`
- **Format:** `[x_min, y_min, x_max, y_max]` in [0, 1000) range

---

### 2. `visualize_bbox_on_image()` (src/utils/visualization.py)

**Expects:** Qwen format `[x_min, y_min, x_max, y_max]`

```python
x1 = int(bbox[0] * width / 1000)   # x_min
y1 = int(bbox[1] * height / 1000)  # y_min
x2 = int(bbox[2] * width / 1000)   # x_max
y2 = int(bbox[3] * height / 1000)  # y_max
```

- **Use for:** Model predictions only
- **Converts:** [0, 1000) normalized coords to pixel coords

---

### 3. `visualize_ground_truth_bbox()` (src/utils/visualization.py)

**Has `format` parameter** to handle different dataset formats:

```python
if format == 'openfoodfacts':
    y_min, x_min, y_max, x_max = bbox  # Unpacks [y,x,y,x]
    gt_x1 = int(x_min * pil_width)      # Reorders to [x,y,x,y]
    gt_y1 = int(y_min * pil_height)
    gt_x2 = int(x_max * pil_width)
    gt_y2 = int(y_max * pil_height)
```

- **Use for:** Ground truth bboxes from dataset
- **Handles:** OpenFoodFacts format `[y_min, x_min, y_max, x_max]`
- **Converts:** [0, 1] normalized coords to pixel coords, reordering axes

---

### 4. `evaluate_model()` (src/training/evaluation.py)

**Converts both formats to the same format for IoU comparison:**

```python
# Ground truth: unpack [y,x,y,x] and reorder to [x,y,x,y]
y_min_gt, x_min_gt, y_max_gt, x_max_gt = ground_truth_bbox
gt_tensor = torch.tensor([[x_min_gt, y_min_gt, x_max_gt, y_max_gt]])

# Prediction: already in [x,y,x,y] format, just normalize from [0,1000) to [0,1]
pred_bbox_norm = [p / 1000.0 for p in pred_bbox]
pred_tensor = torch.tensor([[pred_bbox_norm[0], pred_bbox_norm[1],
                            pred_bbox_norm[2], pred_bbox_norm[3]]])

# Now both are [x_min, y_min, x_max, y_max] in [0,1] range
iou = ops.box_iou(pred_tensor, gt_tensor)
```

- **Use for:** Computing IoU metrics
- **Both converted to:** `[x_min, y_min, x_max, y_max]` normalized [0, 1]

---

## Visual Summary

```
Ground Truth (OpenFoodFacts):     Prediction (Qwen):
[y_min, x_min, y_max, x_max]      [x_min, y_min, x_max, y_max]
        |                                  |
        v                                  v
   visualize_ground_truth_bbox()    visualize_bbox_on_image()
   (converts internally)            (expects Qwen format)
        |                                  |
        v                                  v
   Both draw correctly on image!
```

**For IoU comparison:**
```
Ground Truth --> reorder to [x,y,x,y] --> normalize to [0,1]
Prediction   --> already [x,y,x,y]   --> normalize to [0,1]
                        |
                        v
                 box_iou() works correctly!
```

---

## Quick Reference: Which Function to Use

| Task | Function | Input Format |
|------|----------|--------------|
| Parse model output | `parse_qwen_bbox_output()` | Raw string |
| Visualize prediction | `visualize_bbox_on_image()` | Qwen `[x,y,x,y]` [0,1000) |
| Visualize ground truth | `visualize_ground_truth_bbox()` | OpenFoodFacts `[y,x,y,x]` [0,1] |
| Compute IoU | `evaluate_model()` | Handles both internally |

---

## Why Different Formats?

1. **OpenFoodFacts dataset:** Uses `[y_min, x_min, y_max, x_max]` - this is common in some computer vision datasets where row (y) comes before column (x)

2. **Qwen2-VL model:** Outputs `[x_min, y_min, x_max, y_max]` in [0, 1000) range - this is Qwen's internal normalized coordinate system

The key insight is that **each function knows its expected input format and handles conversion internally**, so you just need to use the right function for the right data source.
