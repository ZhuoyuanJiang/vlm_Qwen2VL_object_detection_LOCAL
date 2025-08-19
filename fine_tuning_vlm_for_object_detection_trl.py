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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="vKadZFQ2IdJb"
# # Fine-Tuning Qwen2-VL-7B for object detection

# %% [markdown] id="rdc7yvCQ7JGf"
# ## 🌟 WHAT?
#
# In this notebook, you will learn how to fine-tune [Qwen2-VL-7B](https://qwenlm.github.io/blog/qwen2-vl/) for for detecting nutrition tables from product images using Hugging Face.
#
# ![Image of nutrition table detection](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*tcy5oCWmHT3jeVN7Lw-FpQ.png)

# %% [markdown] id="NZ84pmL57JGf"
# 💡 You can execute this Jupyter Notebook on a remote machine and then access and interact with it in your local web browser, leveraging the remote machine's computational resources.
# - On remote: jupyter notebook --no-browser --port=8080
# - On local: ssh -L 8080:localhost:8080 ntajbakhsh@workstation

# %% [markdown] id="JATmSI8mcyW2"
# 🚨 **WARNING**: Please note that QWEN2-VL-7B is a relatively large model, requiring significant computational resources for fine-tuning. I recommend using either 2x A6000 or 1x A100 GPUs to ensure sufficient memory and processing power. While I haven't experimented with other GPUs, you're welcome to try alternative options. However, please be aware that other GPUs may not have enough memory to accommodate the model and optimizer states during training.
#
# 🚨 **WARNING**: Training transformers can be significantly more memory-efficient with Flash Attention (FA) compared to traditional attention mechanisms. However, FA support is currently limited to Nvidia's Ampere series of GPUs (A100, A6000, etc.) or better. If you're using an older GPU generation, please note that you'll need to disable FA to avoid error messages. Keep in mind that disabling FA may require using additional GPUs to compensate for the reduced memory efficiency.
#

# %% [markdown] id="gSHmDKNFoqjC"
# # 1. Install Dependencies
#
# Let’s start by installing the essential libraries we’ll need for fine-tuning! 🚀
#

# %% id="GCMhPmFdIGSb"
# !pip install  -U -q git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/trl.git datasets bitsandbytes peft qwen-vl-utils wandb accelerate
# Tested with transformers==4.47.0.dev0, trl==0.12.0.dev0, datasets==3.0.2, bitsandbytes==0.44.1, peft==0.13.2, qwen-vl-utils==0.0.8, wandb==0.18.5, accelerate==1.0.1
# !pip install  matplotlib IPython

# %% [markdown] id="J4pAvoQaOJ1M"
# We’ll also need to install an earlier version of *PyTorch*, as the latest version has an issue that currently prevents this notebook from running correctly. You can learn more about the issue [here](https://github.com/pytorch/pytorch/issues/138340) and consider updating to the latest version once it’s resolved.

# %% id="D8iRteA4oXVj"
# !pip install -q torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# %% [markdown] id="upZjD4bH7JGh"
# # 2. HF Login

# %% [markdown] id="V0-2Lso6wkIh"
# Log in to Hugging Face to upload your fine-tuned model! 🗝️
#
# You’ll need to authenticate with your Hugging Face account to save and share your model directly from this notebook.
#

# %% id="xcL4-bwGIoaR"
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN']) # export your HF_TOKEN first. You can add this to your ~/.bashrc.

# %% [markdown] id="SMZD2Wv57JGi"
# ## Optional Settings for an Improved Jupyter Experience

# %% id="N2BTm9Ug7JGi"
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# %matplotlib inline

# %% [markdown] id="g9QXwbJ7ovM5"
# # 2. Load and Understand Dataset 📁
#
# In this section, you should load the [openfoodfacts/nutrition-table-detection](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection) dataset. This dataset contains product images, the extracted bar codes, and bounding boxes for the nutrition tables.

# %% id="HKIHSAHX7JGi"
# TASK: load the dataset into training and evaluation sets
from datasets import load_dataset
dataset_id = "openfoodfacts/nutrition-table-detection"

# Load the dataset with train and validation splits
ds = load_dataset(dataset_id)
train_dataset = ds['train']
eval_dataset = ds['validation']

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")
print(f"Dataset features: {train_dataset.features}")

# %% id="9J4Y1d-l7JGi"
# TASK: inspect the content of a training example
# Let's look at the first training example
example = train_dataset[0]
print("Example keys:", example.keys())
print("\nBarcode:", example['barcode'])
print("Image size:", example['image'].size if hasattr(example['image'], 'size') else 'N/A')
print("Number of bounding boxes:", len(example['objects']['bbox']))
print("Bounding box:", example['objects']['bbox'][0])
print("Category:", example['objects']['category'][0])

# Display the image with bounding box
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

img = example['image']
draw = ImageDraw.Draw(img)

# Get the bounding box coordinates (normalized to 0-1)
bbox = example['objects']['bbox'][0]
width, height = img.size

# Convert normalized coordinates to pixel coordinates
x_min = bbox[0] * width
y_min = bbox[1] * height
x_max = bbox[2] * width
y_max = bbox[3] * height

# Draw the bounding box
draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title(f"Category: {example['objects']['category'][0]}")
plt.axis('off')
plt.show()

# %% id="eQvNOB-57JGi"
# Q: why the bbox coordinates are between 0 and 1? can you overlay the bbox on the image for one example?
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# The bbox coordinates are normalized to [0, 1] to make them resolution-independent.
# This is a common practice in object detection to handle images of different sizes.

# Let's visualize multiple examples with their bounding boxes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx in range(6):
    example = train_dataset[idx]
    img = example['image'].copy()  # Make a copy to avoid modifying original
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    
    # Draw all bounding boxes for this image
    for bbox, category in zip(example['objects']['bbox'], example['objects']['category']):
        # Convert normalized coordinates to pixel coordinates
        x_min = bbox[0] * width
        y_min = bbox[1] * height
        x_max = bbox[2] * width
        y_max = bbox[3] * height
        
        # Draw the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
        draw.text((x_min, y_min-10), category, fill='red')
    
    axes[idx].imshow(img)
    axes[idx].set_title(f"Image {idx} - Size: {width}x{height}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# %% id="wlr2lSwM7JGi"
# get the histogram of the image sizes
# get the histogram of the #bounding boxes per image - important for finetuning the model
import numpy as np

# Collect image sizes and bbox counts
widths = []
heights = []
bbox_counts = []

for example in train_dataset:
    img = example['image']
    widths.append(img.size[0])
    heights.append(img.size[1])
    bbox_counts.append(len(example['objects']['bbox']))

# Create histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Image widths histogram
axes[0].hist(widths, bins=30, edgecolor='black')
axes[0].set_title('Distribution of Image Widths')
axes[0].set_xlabel('Width (pixels)')
axes[0].set_ylabel('Count')
axes[0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
axes[0].legend()

# Image heights histogram
axes[1].hist(heights, bins=30, edgecolor='black')
axes[1].set_title('Distribution of Image Heights')
axes[1].set_xlabel('Height (pixels)')
axes[1].set_ylabel('Count')
axes[1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
axes[1].legend()

# Bounding boxes per image histogram
axes[2].hist(bbox_counts, bins=range(1, max(bbox_counts)+2), edgecolor='black', align='left')
axes[2].set_title('Distribution of Bounding Boxes per Image')
axes[2].set_xlabel('Number of Bounding Boxes')
axes[2].set_ylabel('Count')
axes[2].set_xticks(range(1, max(bbox_counts)+1))
axes[2].axvline(np.mean(bbox_counts), color='red', linestyle='--', label=f'Mean: {np.mean(bbox_counts):.1f}')
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"Image size stats:")
print(f"  Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}, std={np.std(widths):.0f}")
print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}, std={np.std(heights):.0f}")
print(f"\nBounding boxes per image:")
print(f"  Min: {min(bbox_counts)}, Max: {max(bbox_counts)}, Mean: {np.mean(bbox_counts):.2f}")
print(f"  Most common: {max(set(bbox_counts), key=bbox_counts.count)} boxes (appears {bbox_counts.count(max(set(bbox_counts), key=bbox_counts.count))} times)")

# %% [markdown] id="5nFWDveC7JGi"
# # Understand Model

# %% [markdown] id="l1h_1QI37JGi"
# You should read the Qwen2-VL paper to familiarize yourself with the following:
#
# - **Model Architecture**
# - **Data Processing**
# - **Chat Template**
#
# Next, review the model card and write an inference script for the model using Hugging Face.
#
# Hugging Face provides an abstract API that simplifies usage by hiding many implementation details. While this is convenient, it may leave you with a superficial understanding of the model. To deepen your knowledge, explore the Qwen2-VL code and focus on these key aspects:
#
# - **Understand the input format required by the model:**
#   - Can you create an example input where the user provides two images and one video?
#
# - **Explore `apply_chat_template`:**
#   - Run this function on the example above and analyze the output. What does it do?
#
# - **Understand `process_vision_info`:**
#   - Review the code and determine what this function returns.
#
# - **Examine `processor()`:**
#   - Investigate its functionalities, such as:
#     - Patch-ification
#     - Replicating pad tokens
#     - Text tokenization
#
# - **[Optional] Analyze `model.generate()`'s [forward pass](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L999):**
#   - Understand its operations, including:
#     - Embedding image patches through `PatchEmbed`’s [forward pass](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L272).
#     - Sending patch embeddings to a transformer for feature extraction:
#       - Grasp the concept of 2D RoPE (Rotary Position Embedding).
#       - Pay attention to [forbidden attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L339) when more than one image is provided.
#     - Merging the resulting feature embeddings via [PatchMerger](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L298).
#     - Processing image and text embeddings using the LLM:
#       - Pay special attention to multimodal RoPE.
#
#
#

# %% id="JNIW_O0z7JGi"
# TASK: write an inference function for qwen2-vl
import torch
import os
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

def run_qwen2vl_inference(image_path_or_pil, prompt, model_id="Qwen/Qwen2-VL-7B-Instruct", device="cuda"):
    """
    Run inference with Qwen2-VL model.
    
    Args:
        image_path_or_pil: Either a file path to an image or a PIL Image object
        prompt: Text prompt for the model
        model_id: Model identifier from HuggingFace
        device: Device to run inference on
    
    Returns:
        Generated text response from the model
    """
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    
    # Handle image input - can be path or PIL Image
    if isinstance(image_path_or_pil, str):
        from PIL import Image
        image = Image.open(image_path_or_pil)
    else:
        image = image_path_or_pil
    
    # Create the conversation format expected by Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    
    # Apply chat template to format the conversation
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision information (handles image resizing, patching, etc.)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Prepare inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Deterministic for object detection
        )
    
    # Decode only the generated tokens (excluding the input)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

# Example usage (will be tested in next cell)
print("Inference function defined successfully!")

# %% [markdown] id="jlaMACEb7JGj"
# Test your inference script using this [image](https://t4.ftcdn.net/jpg/01/57/82/05/360_F_157820583_agejYX5XeczPZuWRSCDF2YYeCGwJqUdG.jpg) with the prompt: “Detect the bounding box of the red car.” The model should correctly identify and locate the car in the image, confirming the script’s correctness.

# %% id="xUkdCbsB7JGj"
# TASK: test the inference function
import requests
from PIL import Image
from io import BytesIO

# Download the test image (red car)
test_image_url = "https://t4.ftcdn.net/jpg/01/57/82/05/360_F_157820583_agejYX5XeczPZuWRSCDF2YYeCGwJqUdG.jpg"
response = requests.get(test_image_url)
test_image = Image.open(BytesIO(response.content))

# Display the test image
plt.figure(figsize=(8, 6))
plt.imshow(test_image)
plt.title("Test Image: Red Car")
plt.axis('off')
plt.show()

# Test the inference function
print("Testing Qwen2-VL inference...")
print("Prompt: 'Detect the bounding box of the red car.'")
print("\nModel Response:")

# Uncomment below when you have GPU available
# result = run_qwen2vl_inference(
#     test_image, 
#     "Detect the bounding box of the red car."
# )
# print(result)

# For now, let's show what the expected output format should look like
print("Expected format: <|object_ref_start|>the red car<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>")
print("Where coordinates are normalized to image dimensions (1000x1000 for Qwen2-VL)")

# %% [markdown] id="oosQDLcQ7JGj"
# # Try Qwen2VL without finetuning

# %% [markdown] id="Sy2oO_XE7JGj"
# It’s a good idea to first assess the model’s current capability in detecting the nutrition table without any fine-tuning. This allows for a clear comparison between the model’s performance before and after fine-tuning. To do this, you need to write a function that extracts the bounding box by parsing the model output and then visualize the bounding box on the input image.
#
# Notice that the model’s response will likely follow a different format than wat we saw above for the dog image. Why? One possible explanation is that the nutrition table does not belong to a previously object class seen during the model’s training phase. Additionally, the bounding box coordinates returned by the model are likely inaccurate. This should highlight the necessity of fine-tuning to improve the model’s performance.

# %% id="C5JKCnIA7JGj"
# TASK: write a function to parse model output to extract bounding box coordinates
import re

def parse_qwen_bbox_output(model_output):
    """
    Parse Qwen2-VL model output to extract bounding box coordinates.
    
    Args:
        model_output: String output from the model
        
    Returns:
        Dict with 'object' name and 'bbox' coordinates, or None if parsing fails
    """
    # Pattern for Qwen2-VL detection format:
    # <|object_ref_start|>object name<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    pattern = r'<\|object_ref_start\|>(.+?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    
    matches = re.findall(pattern, model_output)
    
    if not matches:
        # Try alternative format without special tokens (for non-finetuned model)
        # Look for patterns like "bounding box: (x1,y1,x2,y2)" or "[x1,y1,x2,y2]"
        alt_pattern = r'[\[\(]?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]?'
        alt_matches = re.findall(alt_pattern, model_output)
        
        if alt_matches:
            # Take the first match
            coords = alt_matches[0]
            return {
                'object': 'detected object',
                'bbox': [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
            }
        return None
    
    # Parse all detected objects
    results = []
    for match in matches:
        object_name = match[0]
        x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        results.append({
            'object': object_name,
            'bbox': [x1, y1, x2, y2]
        })
    
    return results[0] if len(results) == 1 else results


# TASK: write a function to visualize the bounding boxes on the input image
def visualize_bbox_on_image(image, bbox_data, normalize_coords=True):
    """
    Visualize bounding boxes on an image.
    
    Args:
        image: PIL Image object
        bbox_data: Dict or list of dicts with 'object' and 'bbox' keys
        normalize_coords: If True, bbox coords are in Qwen's 1000x1000 space
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    from PIL import ImageDraw, ImageFont
    
    # Make a copy to avoid modifying original
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    width, height = img_with_bbox.size
    
    # Handle single or multiple bboxes
    if isinstance(bbox_data, dict):
        bbox_data = [bbox_data]
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for idx, data in enumerate(bbox_data):
        if data is None:
            continue
            
        color = colors[idx % len(colors)]
        bbox = data['bbox']
        object_name = data.get('object', 'unknown')
        
        # Convert coordinates if needed
        if normalize_coords:
            # Qwen uses 1000x1000 normalized space
            x1 = int(bbox[0] * width / 1000)
            y1 = int(bbox[1] * height / 1000)
            x2 = int(bbox[2] * width / 1000)
            y2 = int(bbox[3] * height / 1000)
        else:
            x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add label
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = None
        
        label = f"{object_name}"
        if font:
            bbox_label = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox_label, fill=color)
            draw.text((x1, y1-25), label, fill='white', font=font)
        else:
            draw.text((x1, y1-20), label, fill=color)
    
    return img_with_bbox

# Test the parsing function
test_output = "<|object_ref_start|>the red car<|object_ref_end|><|box_start|>(450,380),(650,520)<|box_end|>"
parsed = parse_qwen_bbox_output(test_output)
print("Test parsing:")
print(f"Input: {test_output}")
print(f"Parsed: {parsed}")

# Alternative format test
alt_output = "The bounding box for the object is [100, 200, 300, 400]"
parsed_alt = parse_qwen_bbox_output(alt_output)
print(f"\nAlternative format parsing:")
print(f"Input: {alt_output}")
print(f"Parsed: {parsed_alt}")

# %% [markdown] id="vw_RG5kw7JGj"
# # Data preprocessing

# %% [markdown] id="KGOrsuQo7JGj"
# The dataset requires conversion to be compatible with the Hugging Face (HF) library. Specifically, each sample must be reformatted into the OpenAI conversation format, comprising:
#
# - Roles: system, user, and assistant
# - User input: Provide an image and ask, "Detect the bounding box of the nutrition table."
# - Assistant response: Format compatible with Qwen2-VL's detection question responses
#     * See pages 7 and 43 of this [paper](https://arxiv.org/pdf/2409.12191) and  [Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#more-usage-tips) for tips
#     * Ensure inclusion of class name and bounding box coordinates using the proper special tokens.
#     * Check the expected range of bb coordinates
#     * Pay attention to the order of x,y coordinates as expected by Qwen
#
# Here is an example system prompt:

# %% id="IBvKAlXhI46X"
system_message = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# %% id="XG8NuzqjjbgI"
# Task: write a function to map each sample to a list of 3 dicts (one for each role)

def convert_to_conversation_format(example):
    """
    Convert a dataset example to Qwen2-VL conversation format.
    
    Args:
        example: Dataset sample with 'image', 'objects' containing bbox and category
        
    Returns:
        Dict with 'messages' key containing list of message dicts with roles
    """
    # Extract nutrition table bounding boxes
    bboxes = example['objects']['bbox']
    categories = example['objects']['category']
    
    # Format the assistant response with Qwen2-VL special tokens
    # Convert normalized [0,1] bbox to Qwen's [0,1000] format
    assistant_responses = []
    for bbox, category in zip(bboxes, categories):
        # Convert from [x_min, y_min, x_max, y_max] normalized to Qwen format
        x1 = int(bbox[0] * 1000)
        y1 = int(bbox[1] * 1000)
        x2 = int(bbox[2] * 1000)
        y2 = int(bbox[3] * 1000)
        
        # Format: <|object_ref_start|>object<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        response = f"<|object_ref_start|>{category}<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
        assistant_responses.append(response)
    
    # Combine multiple detections if present
    assistant_text = " ".join(assistant_responses)
    
    # Create conversation format
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example['image']
                },
                {
                    "type": "text",
                    "text": "Detect the bounding box of the nutrition table."
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_text
                }
            ]
        }
    ]
    
    return {"messages": conversation}


# %% [markdown] id="1edUqNGWTtjA"
# Now, let’s format the data using the chatbot structure. This will allow us to set up the interactions appropriately for our model.
#

# %% id="oSHNqk0dkxii"
# Task: apply the function above to all samples in the training and eval datasets


# %% [markdown] id="Sw3b76rawti6"
# # Model finetuning
# **Remove Model and Clean GPU**
#
# Before we proceed with training the model in the next section, let's clear the current variables and clean the GPU to free up resources.
#
#

# %% id="dxkXZuUkvy8j"
import gc
import time

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

clear_memory()

# %% id="Zw9nm6ZP7JGk"
# !nvidia-smi

# %% [markdown] id="yIrR9gP2z90z"
# ## Load the Model for Training with NF4 weights  ⚙️
#
# Next, you need to load the quantized model using [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index). If you want to learn more about quantization, check out [this blog post](https://huggingface.co/blog/merve/quantization) or [this one](https://www.maartengrootendorst.com/blog/quantization/).
#

# %% id="zm_bJRrXsESg"
# TASK: load the NF4 model and processor



# %% [markdown] id="65wfO29isQlX"
# ## Set Up QLoRA and SFTConfig 🚀
#
# Next, you need to configure [QLoRA](https://github.com/artidoro/qlora) for your training setup. QLoRA enables efficient fine-tuning of large language models while significantly reducing the memory footprint compared to traditional methods. Unlike standard LoRA, which reduces memory usage by applying a low-rank approximation, QLoRA takes it a step further by quantizing the model weights. This leads to even lower memory requirements and improved training efficiency, making it an excellent choice for optimizing our model's performance without sacrificing quality.

# %% [markdown] id="S5WRzUJe-P_-"
# 💡 NOTE:
#
# Preparing a model for QLoRA training typically involves three key steps:
#
# - Load the base model in 4-bit (using BitsAndBytesConfig).
#
# - Run prepare_model_for_kbit_training(). 🚨 Understand what this function does.
#
# - Apply LoRA adapters to the target modules.
#
# You can perform these steps manually, or let SFTTrainer handle steps 2 & 3 for you:
#
# - Simply load the model in 4-bit,
#
# - Pass a peft_config to SFTTrainer, which will automatically run prepare_model_for_kbit_training() (for unsharded QLoRA) and attach LoRA adapters. See lines 610 and 625 in [here](https://github.com/huggingface/trl/blob/v0.21.0/trl/trainer/sft_trainer.py)
#
#
#
#

# %% id="ITmkRHWCKYjf"
from peft import LoraConfig, get_peft_model
# Task: create LoRA config and apply LoRA to the model instrance created above


# %% [markdown] id="A1t7Hk_f7JGn"
# Next, you need to create an SFT config for model finetuning. This step is critical for model convergence.
# You should set the following hyper-parameteres among others:
#  - learning_rate
#  - per_device_train_batch_size
#  - gradient_accumulation_steps for better gradient direction estimation
#  - BF16 and TF32 enablement for memory saving and faster compute
#  - gradient_checkpointing for memory saving
#    
# There are other input arguments that you should also set for proper evaluation during model finetuning.

# %% id="SbqX1pQUKaSM"
from trl import SFTConfig
# TASK: create an SFT config


# %% [markdown] id="wjQGt-iZVyef"
# # wandb setup
# If you have wandb account, you can set it up here.
# Let’s connect our notebook to W&B to capture essential information during training.
# Make sure to have set the logging arguments in the SFT config.
#

# %% id="ckVfXDWsoF4Y"
import wandb
# TASK: set up wand.init



# %% [markdown] id="pOUrD9P-y-Kf"
# ## Training the Model 🏃

# %% [markdown] id="ucTUbGURV2_-"
# You should now create a trainer object by instantiating the SFTTrainer class of HF's TRL. For this, you need to provide the training dataset, model, tokenizer, and more important a collate function.
#
# You need a collator function to properly retrieve and batch the data during the training procedure. This function will receive as the input a batch of samples:
#
# [{'role': 'system',
#   'content': [{'type': 'text',
#     'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format. \nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
#  {'role': 'user',
#   'content': [{'type': 'image',
#     'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2592x1944>},
#    {'type': 'text',
#     'text': 'Detect the bounding box of the nutrition table'}]},
#  {'role': 'assistant',
#   'content': [{'type': 'text',
#     'text': '<|object_ref_start|>the nutrition table<|object_ref_end|><|box_start|>(14,57),(991,604)<|box_end|>'}]}]
#     
# [{'role': 'system',
#   'content': [{'type': 'text',
#     'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format. \nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
#  {'role': 'user',
#   'content': [{'type': 'image',
#     'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=306x408>},
#    {'type': 'text',
#     'text': 'Detect the bounding box of the nutrition table'}]},
#  {'role': 'assistant',
#   'content': [{'type': 'text',
#     'text': '<|object_ref_start|>the nutrition table<|object_ref_end|><|box_start|>(147,152),(516,588)<|box_end|>'}]}]
#
# and then performs ops similar to what we did earlier in the inference script:
#   - applying chat template on each sample in the batch -> get formatted prompt
#   - applying process_vision_info on each sample in the batch -> get image pixels
#   - applying processor on formatted prompt and image pixels -> new batch
#   - modify the labels of the new batch by replacing labels correspondings to the following items to -100 (why?)
#     * text pad tokens
#     * <|vision_start|> <|vision_end|> <|image_pad|>
#
# 👉 Check out the TRL official example [scripts]( https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py#L87) for more details.
#

# %% id="pAzDovzylQeZ"
# TASK: Create a data collator to encode text and image pairs


# %% [markdown] id="skbpTuJlV8qN"
# Now, we will define the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), which is a wrapper around the [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class and inherits its attributes and methods. This class simplifies the fine-tuning process by properly initializing the [PeftModel](https://huggingface.co/docs/peft/v0.6.0/package_reference/peft_model) when a [PeftConfig](https://huggingface.co/docs/peft/v0.6.0/en/package_reference/config#peft.PeftConfig) object is provided. By using `SFTTrainer`, we can efficiently manage the training workflow and ensure a smooth fine-tuning experience for our Vision Language Model.
#
#

# %% id="k_jk-U7ULYtA"
from trl import SFTTrainer
# TASK: Create the SFT trainer and launch training


# %% [markdown] id="6yx_sGW42dN3"
# # 5. Testing the Fine-Tuned Model 🔍
#
# Now that we've successfully fine-tuned our Vision Language Model (VLM), it's time to evaluate its performance! In this section, we will test the model using examples from the ChartQA dataset to see how well it answers questions based on chart images. Let's dive in and explore the results! 🚀
#
#

# %% [markdown] id="i0KEPu6qYKqn"
# Let's clean up the GPU memory to ensure optimal performance 🧹

# %% id="Ttx6EK8Uy8t0"
clear_memory()

# %% [markdown] id="HwCTPHsfujn2"
# We will reload the base model using the same pipeline as before, but this we will load the LoRA adpaters into the model too. LoRA adapters should be selected from the saved directory.

# %% id="EFqTNUud2lA7"
# TASK:  Load model, processor, and adapter weights


# %% [markdown] id="pqryChyLWRmR"
# Test the fine-tuned model on the example above, where the model previously struggled to accurately locate the nutrition table.

# %% id="LH6fWLid7JGp"
# TASK: test on the #20 training example


# %% [markdown] id="swibyq5AWctZ"
# Since this sample is drawn from the training set, the model has encountered it during training, which may be seen as a form of cheating. To gain a more comprehensive understanding of the model's performance, you should also evaluate it using the eval dataset. For this, write an evaluation script that measures the IoU metric between the ground truth box and the predicted boundig box.
#

# %% id="czZSBgnoef1E"
# Task: write the eval function. You can use use ops.box_iou
from torchvision import ops


# %% [markdown] id="ATuQ6ZS6eirO" outputId="c3adc0fd-0fdc-4ff4-cc4e-14b4d9039323"
# Do the same evaluation for the model without finetuning.

# %% [markdown] id="iyhR69T3dfLI"
# # 🧑‍🍳 [Optional]  The recipe
# For the best model accuracy, one can first finetune the vision encoder while freezing the LLM. Then, we can use the ckpt above and finetune the model by applying LoRA to vision encoder and QLoRA to LLM.

# %% [markdown] id="ODzQhQ1R1SAR"
# # 🔀 Merge LoRA
# After fine-tuning with LoRA, the adapter weights can be merged back into the base model, effectively eliminating the overhead of LoRA modules during inference. This fusion produces a standalone model suitable for efficient deployment.

# %% [markdown] id="sbDEhQDyqbzK"
# # 🚀 Deployment

# %% [markdown] id="XjtLbVqWqFnE"
# Try to export your trained model (with merged LoRA weights) to vLLM and then deploy into Nvidia triton!

# %% id="b3SH27Iu193x"

# %% [markdown] id="ejJwH5xpcY6K"
# ## Bonus
#
# For Qwen2-VL, implement a custom collate_fn that restricts loss computation to the answer portion only, explicitly excluding the system prompt and question from the loss.

# %% id="JxFyhptrccdr"
