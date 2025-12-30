# Deployment Plan: Qwen2-VL to vLLM + Nvidia Triton

## Goal
Create a deployment notebook (`notebooks/07_deployment_vllm_triton.ipynb`) that exports the trained r4-joint model to vLLM and Nvidia Triton.

## Prerequisites (User Action Required)
- **Transfer checkpoints from vllab13 to vllab14** using rsync:
  ```bash
  # Run from vllab14:
  rsync -avz --progress zhuoyuan@vllab13:/ssd1/zhuoyuan/vlm_outputs/ /ssd1/zhuoyuan/vlm_outputs/
  ```
- After transfer, verify: `ls -la /ssd1/zhuoyuan/vlm_outputs/`
- **Merged model already exists**: `qwen2vl-nutrition-detection-r4-joint-merged` (can skip merge step!)
- Notebook will have configurable paths at the top

## Notebook Structure

### Section 1: Configuration & Setup
- Configurable paths (checkpoint location, output paths)
- Install dependencies: `vllm`, `triton-inference-server` client
- GPU availability check

### Section 2: Merge LoRA Adapter (Optional - Already Done!)
**Note**: The merged model `qwen2vl-nutrition-detection-r4-joint-merged` already exists on vllab13. After rsync, you can skip this section.

If you need to re-merge for any reason:
- Load base model (Qwen2-VL-7B-Instruct) in bf16
- Load r4-joint LoRA adapter
- Merge weights using PEFT's `merge_and_unload()`
- Save merged model to `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged/`
- Verify merged model loads correctly

### Section 3: vLLM Deployment
- **3.1 Launch vLLM Server**
  - Start OpenAI-compatible API server
  - Command: `python -m vllm.entrypoints.openai.api_server --model <merged_model_path> --dtype bfloat16 --trust-remote-code`
  - Run in background subprocess

- **3.2 Test vLLM API**
  - Send test image + prompt via OpenAI client
  - Verify bounding box output format
  - Compare with expected output (sanity check)

- **3.3 Benchmark (Optional)**
  - Throughput test with multiple requests
  - Latency measurement

### Section 4: Triton Deployment
- **4.1 Create Model Repository**
  ```
  model_repository/
  └── qwen2vl_nutrition/
      ├── 1/
      │   └── model.json      # vLLM config
      └── config.pbtxt        # Triton config
  ```

- **4.2 model.json Configuration**
  - model path, dtype (bfloat16), tensor_parallel_size
  - gpu_memory_utilization setting

- **4.3 config.pbtxt Configuration**
  - Backend: vllm
  - Input/output tensor specs
  - Instance configuration

- **4.4 Launch Triton Server**
  - Use NGC container: `tritonserver:24.08-vllm-python-py3`
  - Or install triton-inference-server package
  - Start with model repository

- **4.5 Test Triton API**
  - Use Triton client to send inference requests
  - Verify outputs match vLLM results

### Section 5: Cleanup & Summary
- Stop servers
- Print model paths and deployment commands for reference
- Document API endpoints

## Files to Create

| File | Purpose |
|------|---------|
| `notebooks/07_deployment_vllm_triton.ipynb` | Main deployment notebook |
| `notebooks/07_deployment_vllm_triton.py` | Jupytext-synced Python file |

## Key Dependencies
```
vllm>=0.6.0  # For Qwen2-VL support
tritonclient[all]  # Triton client
openai  # For vLLM OpenAI API testing
```

## Storage Considerations
- Merged model: ~16GB (save to `/ssd1/` not home directory)
- Triton model repository: minimal (just config files, model is referenced by path)

## Key Design Decisions
- **BFloat16 precision** (no serving quantization) - preserves 0.8636 IoU accuracy
- **vLLM first, then Triton** - learn vLLM basics, then add Triton production features
- **Configurable paths** - flexibility across servers (vllab13, vllab15, etc.)

## Background Context

### QLoRA Training vs Serving Quantization
These are completely independent:
- **Training**: Used 4-bit NF4 quantization (QLoRA) to reduce training GPU memory
- **Merged model**: Full bf16 precision (~16GB) after merge_and_unload()
- **Serving**: Keeping bf16 for best accuracy (you have 2×48GB GPUs)

### Model Performance
- Best recipe: r4-joint (4-bit LoRA on both vision & LLM)
- Mean IoU: 0.8636
- Detection rate: 100%

## References
- [vLLM Qwen2.5-VL Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- [Triton vLLM Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html)
- [Triton Quick Deploy Tutorial](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/vLLM/README.html)
