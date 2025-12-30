# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: vlm_Qwen2VL_object_detection
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deploying Qwen2-VL to vLLM and Nvidia Triton
#
# **Purpose**: Export the trained r4-joint model (with merged LoRA weights) to production inference servers.
#
# **What this notebook covers**:
# 1. Configuration and setup
# 2. (Optional) Merge LoRA adapter into base model
# 3. Deploy to vLLM with OpenAI-compatible API
# 4. Deploy to Nvidia Triton with vLLM backend
# 5. Testing and cleanup
#
# **Prerequisites**:
# - Checkpoints transferred from vllab13 to local `/ssd1/zhuoyuan/vlm_outputs/`
# - GPU with sufficient VRAM (~16GB for bf16 model)

# %% [markdown]
# ## Section 1: Configuration & Setup

# %% [markdown]
# ### 1.1 Configuration
#
# **IMPORTANT**: Modify these paths based on your server and checkpoint locations.

# %%
# ============================================================
# CONFIGURATION - Modify these paths for your setup
# ============================================================

# Model paths
BASE_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Where checkpoints are located
ADAPTER_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint"
MERGED_MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"

# vLLM server settings
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
VLLM_MODEL_NAME = "qwen2vl-nutrition"  # Name to use in API calls

# Triton settings
TRITON_MODEL_REPO = "/ssd1/zhuoyuan/triton_model_repository"
TRITON_MODEL_NAME = "qwen2vl_nutrition"
TRITON_PORT = 8001  # gRPC port

# ============================================================
print("Configuration loaded!")
print(f"  Base model: {BASE_MODEL_ID}")
print(f"  Adapter path: {ADAPTER_PATH}")
print(f"  Merged model path: {MERGED_MODEL_PATH}")

# %% [markdown]
# ### 1.2 GPU Configuration
#
# Set up CUDA before importing torch.

# %%
import os
import sys
import subprocess

# Memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_GPUS = 2  # Maximum GPUs to use

print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)


def find_available_gpus(num_gpus_needed=2):
    """Find available GPUs by checking memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        available = []
        print("\nGPU Status:")
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            gpu_idx = int(parts[0])
            mem_used = int(parts[1])
            mem_total = int(parts[2])
            gpu_name = parts[3] if len(parts) > 3 else "Unknown"

            status = "AVAILABLE" if mem_used < 2000 and mem_total > 40000 else "IN USE"
            print(f"  GPU {gpu_idx}: {gpu_name} - {mem_used}MB/{mem_total}MB [{status}]")

            if mem_used < 2000 and mem_total > 40000:
                available.append(gpu_idx)

        return available[:num_gpus_needed] if len(available) >= num_gpus_needed else available

    except Exception as e:
        print(f"Error checking GPUs: {e}")
        return []


available_gpus = find_available_gpus(MAX_GPUS)

if len(available_gpus) > 0:
    gpu_str = ",".join(map(str, available_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"\n✓ Using GPU(s): {gpu_str}")
else:
    print("\n⚠ No available GPUs found! Using whatever is set.")

# %% [markdown]
# ### 1.3 Install Dependencies
#
# Run this cell if you haven't installed vLLM and Triton client yet.

# %%
# Uncomment to install dependencies:
# !pip install vllm>=0.6.0
# !pip install tritonclient[all]
# !pip install openai

# %%
# Check if dependencies are installed
import importlib.util

deps = {
    "vllm": "vLLM (inference server)",
    "openai": "OpenAI client (for testing)",
    "tritonclient": "Triton client",
}

print("Dependency check:")
for module, desc in deps.items():
    spec = importlib.util.find_spec(module)
    status = "✓ Installed" if spec else "✗ Not installed"
    print(f"  {desc}: {status}")

# %% [markdown]
# ### 1.4 Verify Model Paths

# %%
from pathlib import Path

print("=" * 60)
print("MODEL PATH VERIFICATION")
print("=" * 60)

# Check adapter path
adapter_path = Path(ADAPTER_PATH)
if adapter_path.exists():
    adapter_config = adapter_path / "adapter_config.json"
    if adapter_config.exists():
        print(f"✓ LoRA adapter found at: {ADAPTER_PATH}")
        # List contents
        files = list(adapter_path.iterdir())
        print(f"  Files: {len(files)} items")
    else:
        print(f"⚠ Directory exists but no adapter_config.json: {ADAPTER_PATH}")
else:
    print(f"✗ Adapter not found: {ADAPTER_PATH}")

# Check merged model path
merged_path = Path(MERGED_MODEL_PATH)
if merged_path.exists():
    safetensors = list(merged_path.glob("*.safetensors"))
    if safetensors:
        total_size = sum(f.stat().st_size for f in safetensors) / 1024**3
        print(f"✓ Merged model found at: {MERGED_MODEL_PATH}")
        print(f"  Model size: {total_size:.2f} GB ({len(safetensors)} files)")
    else:
        print(f"⚠ Directory exists but no safetensor files: {MERGED_MODEL_PATH}")
else:
    print(f"✗ Merged model not found: {MERGED_MODEL_PATH}")
    print("  → Run Section 2 to merge LoRA adapter into base model")

# %% [markdown]
# ## Section 2: Merge LoRA Adapter (Optional)
#
# **Skip this section if** `qwen2vl-nutrition-detection-r4-joint-merged` already exists!
#
# This section merges the LoRA adapter weights into the base model to create a standalone model
# that doesn't require PEFT at inference time.

# %% [markdown]
# ### 2.1 Check if Merge is Needed

# %%
skip_merge = Path(MERGED_MODEL_PATH).exists() and list(Path(MERGED_MODEL_PATH).glob("*.safetensors"))

if skip_merge:
    print("✓ Merged model already exists! Skipping Section 2.")
    print(f"  Path: {MERGED_MODEL_PATH}")
else:
    print("⚠ Merged model not found. Run the cells below to merge.")

# %% [markdown]
# ### 2.2 Load Base Model and LoRA Adapter
#
# Only run this cell if merge is needed.

# %%
# Skip if already merged
if not skip_merge:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from peft import PeftModel

    print("Loading base model in bfloat16 (for merging)...")
    print("This may take a few minutes...")

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Base model loaded: {base_model.get_memory_footprint() / 1024**3:.2f} GB")

    print("\nLoading LoRA adapter...")
    model_with_lora = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✓ LoRA adapter loaded")
else:
    print("Skipping - model already merged")

# %% [markdown]
# ### 2.3 Merge and Save

# %%
# Skip if already merged
if not skip_merge:
    print("Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()
    print(f"✓ Merge complete! Memory: {merged_model.get_memory_footprint() / 1024**3:.2f} GB")

    print("\nLoading processor...")
    processor = Qwen2VLProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )

    print(f"\nSaving merged model to: {MERGED_MODEL_PATH}")
    merged_path = Path(MERGED_MODEL_PATH)
    merged_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(MERGED_MODEL_PATH)
    processor.save_pretrained(MERGED_MODEL_PATH)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    for f in sorted(merged_path.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Free memory
    del base_model, model_with_lora, merged_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Skipping - model already merged")

# %% [markdown]
# ## Section 3: vLLM Deployment
#
# Deploy the merged model using vLLM's OpenAI-compatible API server.
#
# **vLLM provides**:
# - OpenAI-compatible `/v1/chat/completions` endpoint
# - PagedAttention for efficient memory management
# - Continuous batching for high throughput

# %% [markdown]
# ### 3.1 Launch vLLM Server
#
# We'll launch vLLM as a background subprocess. The server provides an OpenAI-compatible API.

# %% [markdown]
# ### What is "OpenAI-compatible API"?
#
# It means vLLM mimics OpenAI's API format exactly, so any code written for OpenAI's ChatGPT API works with vLLM without changes.
#
# #### Example: The Same Code Works for Both
#
# **Calling OpenAI's API (ChatGPT):**
# ```python
# from openai import OpenAI
#
# client = OpenAI(
#     api_key="sk-xxxxx",  # Your OpenAI API key
#     base_url="https://api.openai.com/v1"  # OpenAI's server
# )
#
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "Hello!"}]
# )
# print(response.choices[0].message.content)
# ```
#
# **Calling vLLM (Your Local Model) - Same Code!**
# ```python
# from openai import OpenAI
#
# client = OpenAI(
#     api_key="dummy",  # vLLM doesn't need a real key
#     base_url="http://localhost:8000/v1"  # Your vLLM server
# )
#
# response = client.chat.completions.create(
#     model="qwen2vl-nutrition",  # Your model name
#     messages=[{"role": "user", "content": "Hello!"}]
# )
# print(response.choices[0].message.content)
# ```
#
# ### Why This Matters
#
# | Benefit             | Explanation                                                                  |
# |---------------------|------------------------------------------------------------------------------|
# | Drop-in replacement | Switch from OpenAI to your model by changing 2 lines (base_url + model name) |
# | Existing tools work | Any app built for ChatGPT API works with vLLM                                |
# | Same documentation  | OpenAI's API docs apply to vLLM                                              |
# | Industry standard   | Many inference servers now support this format                               |

# %%
import subprocess
import time
import signal
import requests

# Build the vLLM command
vllm_cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MERGED_MODEL_PATH,
    "--served-model-name", VLLM_MODEL_NAME,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    "--dtype", "bfloat16",
    "--trust-remote-code",
    "--max-model-len", "4096",  # Limit context length for memory
]

print("vLLM Launch Command:")
print(" ".join(vllm_cmd))
print()
print("To launch manually in a terminal, run:")
print(f"  cd {Path.cwd()}")
print(f"  {' '.join(vllm_cmd)}")

# %% [markdown]
# ### 3.1.1 Launch vLLM (Background Process)
#
# **Option A**: Run this cell to launch vLLM as a background subprocess.
#
# **Option B**: Open a new terminal and run the command printed above.

# %%
# Launch vLLM server as background process
# Set to True to launch, False to skip (if running manually)
LAUNCH_VLLM = False  # Change to True to launch from notebook

if LAUNCH_VLLM:
    print("Launching vLLM server...")
    vllm_process = subprocess.Popen(
        vllm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"✓ vLLM process started (PID: {vllm_process.pid})")

    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_wait = 120  # seconds
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"http://localhost:{VLLM_PORT}/health")
            if resp.status_code == 200:
                print(f"✓ Server ready! (took {time.time() - start:.1f}s)")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    else:
        print(f"⚠ Server not ready after {max_wait}s")
else:
    print("LAUNCH_VLLM is False - launch vLLM manually in a terminal:")
    print(f"  {' '.join(vllm_cmd)}")
    vllm_process = None

# %% [markdown]
# ### 3.2 Test vLLM API
#
# Send a test request with an image to verify the deployment.

# %%
import base64
from openai import OpenAI
from PIL import Image
import io

# vLLM endpoint
client = OpenAI(
    base_url=f"http://localhost:{VLLM_PORT}/v1",
    api_key="dummy",  # vLLM doesn't need a real key
)

# Test prompt (same as training)
SYSTEM_PROMPT = """You are a nutrition label detector. Your task is to identify nutrition tables/panels in food product images.

When you detect a nutrition table, output its location using this exact format:
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates are in [0,1000) range where (0,0) is top-left.
If no nutrition table is found, say "No nutrition table detected."
"""

USER_PROMPT = "Detect the bounding box coordinates for the nutrition facts table in this image."


def encode_image_to_base64(image_path):
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_vllm_inference(image_path: str):
    """Test vLLM inference with an image."""
    print(f"Testing with image: {image_path}")

    # Encode image
    image_b64 = encode_image_to_base64(image_path)

    # Build message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )

        result = response.choices[0].message.content
        print(f"\nModel output:\n{result}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None

# %% [markdown]
# ### 3.2.1 Run Test Inference
#
# Modify the image path to test with your own image.

# %%
# Test with a sample image
# You can use any image with a nutrition table

# Option 1: Use a test image from your dataset
# test_image = "/path/to/your/test/image.jpg"

# Option 2: Download a sample image
print("To test, either:")
print("1. Set test_image = '/path/to/your/nutrition_label_image.jpg'")
print("2. Or use the dataset:")
print("   from datasets import load_dataset")
print("   ds = load_dataset('openfoodfacts/nutrition-table-detection', split='test')")
print("   ds[0]['image'].save('/tmp/test_nutrition.jpg')")
print("   test_image = '/tmp/test_nutrition.jpg'")

# Uncomment and modify to test:
# test_image = "/tmp/test_nutrition.jpg"
# result = test_vllm_inference(test_image)

# %%
# # Uncomment to use test image from dataset
# # Load test image from dataset
# from datasets import load_dataset

# ds = load_dataset("openfoodfacts/nutrition-table-detection", split="test")
# ds[0]['image'].save('/tmp/test_nutrition.jpg')

# test_image = "/tmp/test_nutrition.jpg"
# result = test_vllm_inference(test_image)

# %% [markdown]
# ### 3.3 API Usage Examples
#
# Here are different ways to call the vLLM API.

# %%
# Example 1: Using curl (from terminal)
curl_example = f"""
# Test health endpoint
curl http://localhost:{VLLM_PORT}/health

# Test chat completion (text only)
curl http://localhost:{VLLM_PORT}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{VLLM_MODEL_NAME}",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 50
  }}'
"""

print("curl Examples:")
print(curl_example)

# %%
# Example 2: Using Python requests
requests_example = f'''
import requests
import base64

# Health check
resp = requests.get("http://localhost:{VLLM_PORT}/health")
print(f"Health: {{resp.json()}}")

# Chat completion with image
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:{VLLM_PORT}/v1/chat/completions",
    json={{
        "model": "{VLLM_MODEL_NAME}",
        "messages": [
            {{"role": "system", "content": "You are a nutrition label detector..."}},
            {{"role": "user", "content": [
                {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{img_b64}}"}}}},
                {{"type": "text", "text": "Detect the bounding box..."}}
            ]}}
        ],
        "max_tokens": 256
    }}
)
print(resp.json())
'''

print("Python requests Example:")
print(requests_example)

# %%
# This shows the **actual raw API call** (not using wrapper function).
# We use a different test image to demonstrate.


import requests
import base64
from datasets import load_dataset

# Load a DIFFERENT test image (index 5 instead of 0)
print("Loading test image (sample #5 from dataset)...")
ds = load_dataset("openfoodfacts/nutrition-table-detection", split="test")
ds[5]['image'].save('/tmp/test_api_example.jpg')
print("Saved to /tmp/test_api_example.jpg")

# Encode to base64
with open('/tmp/test_api_example.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

print(f"Image encoded: {len(img_b64)} characters")

# Make the actual API call
print("\nSending request to vLLM API...")
response = requests.post(
    f"http://localhost:{VLLM_PORT}/v1/chat/completions",
    json={
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": USER_PROMPT}
            ]}
        ],
        "max_tokens": 256,
        "temperature": 0.0
    },
    timeout=60
)

# Show the result
print("\n" + "="*50)
print("API Response:")
print("="*50)
result = response.json()
print(f"Model output: {result['choices'][0]['message']['content']}")

# %% [markdown]
# ### 3.3.1 Real API Call Example
#

# %%
# This shows the **actual raw API call** (not using wrapper function).
# We use a different test image to demonstrate.

import requests
import base64
from datasets import load_dataset

# Load a DIFFERENT test image (index 5 instead of 0)
print("Loading test image (sample #5 from dataset)...")
ds = load_dataset("openfoodfacts/nutrition-table-detection", split="test")
ds[5]['image'].save('/tmp/test_api_example.jpg')
print("Saved to /tmp/test_api_example.jpg")

# Encode to base64
with open('/tmp/test_api_example.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

print(f"Image encoded: {len(img_b64)} characters")

# Make the actual API call
print("\nSending request to vLLM API...")
response = requests.post(
    f"http://localhost:{VLLM_PORT}/v1/chat/completions",
    json={
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": USER_PROMPT}
            ]}
        ],
        "max_tokens": 256,
        "temperature": 0.0
    },
    timeout=60
)

# Show the result
print("\n" + "="*50)
print("API Response:")
print("="*50)
result = response.json()
print(f"Model output: {result['choices'][0]['message']['content']}")


# %% [markdown]
# ### 3.4 Stop vLLM Server
#
# Run this cell when you're done testing vLLM.

# %%
def stop_vllm():
    """Stop the vLLM server if running."""
    if 'vllm_process' in globals() and vllm_process is not None:
        print("Stopping vLLM server...")
        vllm_process.terminate()
        vllm_process.wait(timeout=10)
        print("✓ vLLM server stopped")
    else:
        print("vLLM not launched from notebook")
        print("If running manually, press Ctrl+C in the terminal")


# Uncomment to stop:
# stop_vllm()

# %% [markdown]
# ## Section 4: Triton Deployment
#
# Deploy to Nvidia Triton Inference Server with vLLM backend.
#
# **Triton adds**:
# - Metrics endpoint (Prometheus-compatible)
# - Model versioning
# - Multi-model serving
# - HTTP/gRPC endpoints
# - Ensemble support

# %% [markdown]
# ### 4.1 Create Model Repository
#
# Triton requires a specific directory structure.

# %%
import json
from pathlib import Path

# Create model repository structure
repo_path = Path(TRITON_MODEL_REPO)
model_path = repo_path / TRITON_MODEL_NAME / "1"
model_path.mkdir(parents=True, exist_ok=True)

print(f"Creating Triton model repository at: {repo_path}")

# %% [markdown]
# ### 4.2 Create model.json
#
# This file configures the vLLM engine.

# %%
# vLLM configuration for Triton
model_json = {
    "model": MERGED_MODEL_PATH,
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,  # Set to 2 if using 2 GPUs
    "gpu_memory_utilization": 0.8,  # Use 80% of GPU memory
    "max_model_len": 4096,
    "trust_remote_code": True,
}

model_json_path = model_path / "model.json"
with open(model_json_path, "w") as f:
    json.dump(model_json, f, indent=2)

print(f"Created: {model_json_path}")
print(json.dumps(model_json, indent=2))

# %% [markdown]
# ### 4.3 Create config.pbtxt
#
# This file configures Triton's handling of the model.

# %%
config_pbtxt = f'''name: "{TRITON_MODEL_NAME}"
backend: "vllm"

# Maximum batch size (0 = batching handled by vLLM)
max_batch_size: 0

# Input configuration
input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "image"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }}
]

# Output configuration
output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

# Instance configuration
instance_group [
  {{
    count: 1
    kind: KIND_MODEL
  }}
]
'''

config_path = repo_path / TRITON_MODEL_NAME / "config.pbtxt"
with open(config_path, "w") as f:
    f.write(config_pbtxt)

print(f"Created: {config_path}")
print(config_pbtxt)

# %% [markdown]
# ### 4.4 Verify Model Repository Structure

# %%
print("Model Repository Structure:")
print(f"{repo_path}/")
for p in sorted(repo_path.rglob("*")):
    rel = p.relative_to(repo_path)
    indent = "  " * len(rel.parts)
    if p.is_file():
        size = p.stat().st_size
        print(f"{indent}{p.name} ({size} bytes)")
    else:
        print(f"{indent}{p.name}/")

# %% [markdown]
# ### 4.5 Launch Triton Server
#
# **Option A**: Use NGC Docker container (recommended)
#
# **Option B**: Use local installation

# %%
# Option A: Docker command (recommended)
docker_cmd = f"""
# Pull the Triton container with vLLM backend
docker pull nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3

# Run Triton
docker run --gpus all --rm -it \\
  --shm-size=4G \\
  -p 8000:8000 \\
  -p {TRITON_PORT}:8001 \\
  -p 8002:8002 \\
  -v {MERGED_MODEL_PATH}:{MERGED_MODEL_PATH}:ro \\
  -v {repo_path}:/models \\
  nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \\
  tritonserver --model-repository=/models
"""

print("Docker Launch Command:")
print(docker_cmd)

# %%
# Option B: Local command (if tritonserver is installed)
local_cmd = f"""
tritonserver --model-repository={repo_path}
"""

print("Local Launch Command:")
print(local_cmd)

# %% [markdown]
# ### 4.6 Test Triton API
#
# Test the deployed model using Triton client.

# %%
# Note: Run this after Triton server is running

try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient

    triton_available = True
except ImportError:
    print("⚠ tritonclient not installed. Run: pip install tritonclient[all]")
    triton_available = False


def test_triton_inference(prompt: str, image_b64: str = None):
    """Test Triton inference."""
    if not triton_available:
        print("tritonclient not available")
        return None

    try:
        # Use HTTP client
        client = httpclient.InferenceServerClient(url=f"localhost:8000")

        # Check server health
        if not client.is_server_live():
            print("⚠ Triton server not live")
            return None

        print(f"✓ Triton server is live")

        # Create input
        text_input = httpclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=np.object_))

        inputs = [text_input]

        if image_b64:
            image_input = httpclient.InferInput("image", [1], "BYTES")
            image_input.set_data_from_numpy(np.array([image_b64.encode()], dtype=np.object_))
            inputs.append(image_input)

        # Create output
        output = httpclient.InferRequestedOutput("text_output")

        # Infer
        response = client.infer(
            model_name=TRITON_MODEL_NAME,
            inputs=inputs,
            outputs=[output],
        )

        result = response.as_numpy("text_output")
        print(f"Output: {result}")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


# Test example
# test_triton_inference("Detect the nutrition table in this image", image_b64="...")

# %% [markdown]
# ### 4.7 Triton API Examples

# %%
triton_curl_example = f"""
# Check server health
curl -v http://localhost:8000/v2/health/live

# Check model status
curl http://localhost:8000/v2/models/{TRITON_MODEL_NAME}

# Get metrics (Prometheus format)
curl http://localhost:8002/metrics

# Inference request
curl -X POST http://localhost:8000/v2/models/{TRITON_MODEL_NAME}/infer \\
  -H "Content-Type: application/json" \\
  -d '{{
    "inputs": [
      {{
        "name": "text_input",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["Detect the nutrition table."]
      }}
    ]
  }}'
"""

print("Triton curl Examples:")
print(triton_curl_example)

# %% [markdown]
# ## Section 5: Summary & Cleanup

# %% [markdown]
# ### 5.1 Deployment Summary

# %%
print("=" * 60)
print("DEPLOYMENT SUMMARY")
print("=" * 60)

print("\n📁 Model Paths:")
print(f"  Base Model: {BASE_MODEL_ID}")
print(f"  LoRA Adapter: {ADAPTER_PATH}")
print(f"  Merged Model: {MERGED_MODEL_PATH}")

print(f"\n🚀 vLLM Deployment:")
print(f"  Endpoint: http://localhost:{VLLM_PORT}/v1/chat/completions")
print(f"  Model Name: {VLLM_MODEL_NAME}")
print(f"  Health: http://localhost:{VLLM_PORT}/health")

print(f"\n🔧 Triton Deployment:")
print(f"  Model Repository: {TRITON_MODEL_REPO}")
print(f"  HTTP API: http://localhost:8000/v2/models/{TRITON_MODEL_NAME}")
print(f"  gRPC: localhost:{TRITON_PORT}")
print(f"  Metrics: http://localhost:8002/metrics")

print("\n📋 Quick Commands:")
print(f"  Launch vLLM:")
print(f"    {' '.join(vllm_cmd)}")
print(f"\n  Launch Triton (Docker):")
print(f"    docker run --gpus all --rm -p 8000:8000 -v {repo_path}:/models \\")
print(f"      nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 tritonserver --model-repository=/models")

# %% [markdown]
# ### 5.2 Cleanup

# %%
def cleanup():
    """Stop servers and free resources."""
    # Stop vLLM if running
    if 'vllm_process' in globals() and vllm_process is not None:
        print("Stopping vLLM server...")
        vllm_process.terminate()
        vllm_process.wait(timeout=10)
        print("✓ vLLM stopped")

    # Free GPU memory
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU memory cleared")
    except ImportError:
        pass

    print("\nCleanup complete!")
    print("Note: Triton must be stopped manually (Ctrl+C or docker stop)")


# Uncomment to cleanup:
# cleanup()

# %%
cleanup()

# %% [markdown]
# ### 5.3 Next Steps
#
# 1. **Production Deployment**:
#    - Use Docker Compose for orchestration
#    - Add load balancer (nginx) for multiple replicas
#    - Set up monitoring with Prometheus/Grafana
#
# 2. **Performance Optimization**:
#    - Enable tensor parallelism for multi-GPU
#    - Tune `gpu_memory_utilization` and `max_model_len`
#    - Consider FP8 quantization if memory constrained
#
# 3. **API Integration**:
#    - Build client SDK for your application
#    - Add authentication/rate limiting
#    - Implement batch inference pipeline

# %% [markdown]
# ---
#
# **Notebook**: `07_deployment_vllm_triton.ipynb`
#
# **Author**: Generated for Qwen2-VL Nutrition Detection Project
#
# **References**:
# - [vLLM Documentation](https://docs.vllm.ai/)
# - [Triton vLLM Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html)
# - [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
