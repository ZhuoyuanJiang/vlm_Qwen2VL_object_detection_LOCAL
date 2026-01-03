# vLLM Model Repetition Bug - Debug Summary

## Context
- **Model**: Qwen2-VL 7B fine-tuned for nutrition table detection
- **Merged model path**: `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged`
- **Server**: vllab8 with vLLM 0.13.0 + PyTorch 2.9.0+cu129
- **Conda env**: `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving`

## The Problem

When serving the model with vLLM, **the model repeats its output indefinitely** until `max_tokens` is reached:

```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)<|box_end|>
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)<|box_end|>
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(73...
```

## Expected Output Format

With `skip_special_tokens=False`:
```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
```

With `skip_special_tokens=True`:
```
nutrition-table(x1,y1),(x2,y2)
```

## What We Tried with vLLM

| Approach | Result |
|----------|--------|
| `stop: ["<|box_end|>"]` | Works but cuts off `<|box_end|>` from output |
| `stop: ["<|im_end|>"]` | Still repeats |
| `stop_token_ids: [151645, 151643]` | Still repeats |
| No stop tokens | Repeats until max_tokens |

## Model Configuration (Verified)

```python
eos_token: <|im_end|>
eos_token_id: 151645
generation_config.eos_token_id: [151645, 151643]

# Special tokens exist in tokenizer:
<|object_ref_start|>: 151646
<|object_ref_end|>: 151647
<|box_start|>: 151648
<|box_end|>: 151649
```

## vLLM Serve Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --port 8000
```

## Test Script

```python
import base64
import requests
from datasets import load_dataset

ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")
ds[0]['image'].save('/tmp/test_nutrition.jpg')

with open('/tmp/test_nutrition.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen2vl-nutrition",
        "messages": [
            {"role": "system", "content": "You are a Vision Language Model specialized in interpreting visual data from product images. Your task is to analyze the provided product images and detect the nutrition tables in a certain format. Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Detect the bounding box coordinates for the nutrition facts table."}
            ]}
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stop": ["<|box_end|>"],  # Current workaround
        "skip_special_tokens": False
    }
)
```

## Investigation Notes

The model's `generation_config.json` has correct EOS tokens configured. Further investigation needed to determine root cause of repetition behavior.

## Current Workaround

Using `stop: ["<|box_end|>"]` prevents repetition but requires adjusting the parsing logic since `<|box_end|>` is not included in the output.

## Training System Prompt

The model was trained with this system message:
```
You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.
```

## Files to Reference

- Test script: `scripts/test_vllm_api.py` - minimal API test
- Detailed test script: `scripts/test_vllm_with_visualization.py` - same API call + local post-processing (parsing, visualization)
- Inference function: `fine_tuning_vlm_for_object_detection_trl.py` (search for `run_qwen2vl_inference`)
- Inference reference: `notebooks/02_model_understanding.py`
