# Triton Model Repository for Qwen2-VL Nutrition Detection

This directory contains the Triton Inference Server configuration for deploying the fine-tuned Qwen2-VL model.

## Directory Structure

```
triton_model_repository/
├── README.md                              # This file
├── PATH_MAPPING.md                        # Path mapping for cloud deployment
│
├── qwen2vl_nutrition_bf16/                # BF16 full-precision model (GPU 0)
│   ├── config.pbtxt
│   └── 1/
│       └── model.json
│
└── qwen2vl_nutrition_gptq_int4/           # GPTQ INT4 quantized model (GPU 1)
    ├── config.pbtxt
    └── 1/
        └── model.json
```

## Models

| Model | Endpoint | GPU | Size | Use Case |
|-------|----------|-----|------|----------|
| `qwen2vl_nutrition_bf16` | `/v2/models/qwen2vl_nutrition_bf16/infer` | 0 | ~15.5 GB | Accuracy baseline |
| `qwen2vl_nutrition_gptq_int4` | `/v2/models/qwen2vl_nutrition_gptq_int4/infer` | 1 | ~6.5 GB | Production (faster) |

## Before Deployment: Update Paths!

The `model.json` files contain placeholder paths. Update them for your cloud server:

```json
// BF16 model.json
"model": "/models/qwen2vl-nutrition-detection-r4-joint-merged"

// GPTQ INT4 model.json
"model": "/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"
```

## Quick Start (Cloud Server)

### 1. Transfer Files

```bash
# Transfer BF16 model weights (~15GB)
rsync -avz --progress \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  user@cloud:/models/

# Transfer GPTQ INT4 model weights (~7GB)
rsync -avz --progress \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  user@cloud:/models/

# Transfer this config directory
rsync -avz --progress \
  triton_model_repository/ \
  user@cloud:/workspace/triton_model_repository/
```

### 2. Start Triton Server (Both Models)

```bash
docker run --gpus all --rm -it \
    --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /models/qwen2vl-nutrition-detection-r4-joint-merged:/models/qwen2vl-nutrition-detection-r4-joint-merged:ro \
    -v /models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
    -v /workspace/triton_model_repository:/models/triton_repo \
    nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \
    tritonserver --model-repository=/models/triton_repo
```

### 3. Test Health

```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/models/qwen2vl_nutrition_bf16
curl http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4
```

### 4. Test Inference

```python
import base64
import requests

# Encode image
with open("test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send request to GPTQ INT4 model
payload = {
    "inputs": [
        {"name": "text_input", "shape": [1], "datatype": "BYTES",
         "data": ["Detect the nutrition facts table in this image."]},
        {"name": "image", "shape": [1], "datatype": "BYTES",
         "data": [image_b64]},
        {"name": "sampling_parameters", "shape": [1], "datatype": "BYTES",
         "data": ['{"temperature": 0, "max_tokens": 100}']},
        {"name": "stream", "shape": [1], "datatype": "BOOL",
         "data": [False]}
    ]
}

# Choose model: bf16 or gptq_int4
model_name = "qwen2vl_nutrition_gptq_int4"  # or "qwen2vl_nutrition_bf16"

response = requests.post(
    f"http://localhost:8000/v2/models/{model_name}/infer",
    json=payload
)
print(response.json())
```

## A/B Testing

With both models loaded, you can compare outputs:

```python
# Same image, same prompt -> compare outputs
bf16_response = requests.post(".../qwen2vl_nutrition_bf16/infer", json=payload)
gptq_response = requests.post(".../qwen2vl_nutrition_gptq_int4/infer", json=payload)

# Compare bounding box outputs
print("BF16:", bf16_response.json())
print("GPTQ:", gptq_response.json())
```

## API Endpoints

| Endpoint | Port | Purpose |
|----------|------|---------|
| HTTP | 8000 | REST API |
| gRPC | 8001 | gRPC API |
| Metrics | 8002 | Prometheus metrics |

## Important Notes

1. **Use `/infer` not `/generate`**: For VLMs with image input, use the inference endpoint
2. **Image format**: Base64-encoded JPEG/PNG
3. **Decoupled mode**: The backend uses streaming protocol even for non-streaming requests
4. **Chat template**: Apply chat template client-side or use raw prompts that match training format
5. **GPU requirement**: Need 2 GPUs to run both models simultaneously (or use explicit model control)

## Troubleshooting

### Model fails to load
- Check path in `model.json` matches mounted location
- Verify GPU has enough memory (BF16 needs ~16GB, GPTQ needs ~8GB)
- Check `--shm-size=4G` is set in Docker run command

### Empty or incorrect outputs
- Verify prompt format matches training format
- Check `sampling_parameters` (temperature, max_tokens)
- Test with same image/prompt that works on standalone vLLM

### Out of memory with both models
- Use `--model-control-mode=explicit` and load one model at a time
- Or deploy on a server with 2+ GPUs
