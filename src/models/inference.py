"""
Inference Functions for Qwen2-VL Model

QUICK REFERENCE:
- WHEN: For testing model before/after training
- PURPOSE: Run inference and parse outputs
- RUNS: On-demand for inference (not during training)
"""

import torch
import re
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


def run_qwen2vl_inference(image_path_or_pil, prompt, model_id="Qwen/Qwen2-VL-7B-Instruct",
                          device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run inference with Qwen2-VL model.

    WHEN: Test model on single image (before/after fine-tuning)

    INPUT:
        image_path_or_pil: Either a file path (str) or PIL Image object
        prompt: Text prompt for the model (str)
        model_id: HuggingFace model identifier (str)
        device: Device to run on ("cuda" or "cpu")

    OUTPUT:
        str: Generated text response from the model

    OUTPUT FORMAT (with skip_special_tokens=True):
        "object name(x1,y1),(x2,y2)"
        - Coordinates are in Qwen's normalized [0, 1000] space
        - Example: "nutrition table(13,60),(984,989)"

    Example Usage:
        >>> from PIL import Image
        >>> image = Image.open("product.jpg")
        >>> result = run_qwen2vl_inference(image, "Detect the nutrition table")
        >>> print(result)  # "nutrition table(x1,y1),(x2,y2)"

    Why This Function Exists:
        - Complete pipeline for testing models
        - Handles both file paths and PIL Images
        - Consistent format for all inference tasks
        - Easy to compare pre-trained vs fine-tuned

    Pipeline Steps:
        1. Load model and processor
        2. Handle image input (path or PIL)
        3. Format as conversation (messages list)
        4. Apply chat template
        5. Process vision information
        6. Generate with model
        7. Decode generated tokens only
    """
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Original training precision
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = Qwen2VLProcessor.from_pretrained(
        model_id,
        min_pixels=256*28*28,    # ~200k pixels (reduced from 3k default)
        max_pixels=1280*28*28,   # ~1M pixels (reduced from 12.8M default!)
        trust_remote_code=True
    )

    # Set model to evaluation mode for consistent inference
    model.eval()

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
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template to format the conversation
    text = processor.apply_chat_template(
        messages,
        tokenize=False,  # Return text, not token IDs
        add_generation_prompt=True  # Add assistant token to prompt response
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
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def parse_qwen_bbox_output(model_output):
    """
    Parse Qwen2-VL model output to extract bounding box coordinates.

    WHEN: After running inference, to get structured bbox data

    INPUT:
        model_output (str): Raw text output from model

    OUTPUT:
        dict or list of dicts or None:
        - Single detection: {'object': 'car', 'bbox': [x1, y1, x2, y2]}
        - Multiple detections: [{'object': 'car', 'bbox': [...]}, ...]
        - Parse failure: None

    SUPPORTED FORMATS:
        Format 1 (WITH special tokens, skip_special_tokens=False):
            "<|object_ref_start|>object name<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>"

        Format 2 (WITHOUT special tokens, skip_special_tokens=True):
            "object name(x1,y1),(x2,y2)"

    COORDINATE SPACE:
        - Qwen2-VL uses normalized [0, 1000] space
        - To convert to pixels: pixel_x = (x * image_width) / 1000

    Example Usage:
        >>> output = "the red car(358,571),(492,943)"
        >>> result = parse_qwen_bbox_output(output)
        >>> print(result)
        {'object': 'the red car', 'bbox': [358, 571, 492, 943]}

        >>> # Multiple detections
        >>> output = "table 1(10,20),(100,200) and table 2(500,600),(900,900)"
        >>> result = parse_qwen_bbox_output(output)
        >>> print(result)
        [{'object': 'table 1', 'bbox': [10, 20, 100, 200]},
         {'object': 'table 2', 'bbox': [500, 600, 900, 900]}]

    Why This Function Exists:
        - Model outputs text, we need structured data
        - Handles multiple output formats
        - Robust to extra text in response
        - Returns consistent format for visualization
    """
    # Remove <|im_end|> token if present (appears when skip_special_tokens=False)
    model_output = model_output.replace('<|im_end|>', '').strip()

    # Pattern 1: WITH special tokens (skip_special_tokens=False)
    # Format: <|object_ref_start|>object name<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    pattern_with_tokens = r'<\|object_ref_start\|>(.+?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern_with_tokens, model_output)

    if matches:
        # Can handle multiple detections
        results = []
        for match in matches:
            object_name = match[0]
            x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            results.append({
                'object': object_name,
                'bbox': [x1, y1, x2, y2]  # Qwen outputs in [x,y,x,y] format
            })
        return results[0] if len(results) == 1 else results

    # Pattern 2: WITHOUT special tokens (skip_special_tokens=True)
    # Format: "object name(x1,y1),(x2,y2)" - may have space before parenthesis
    pattern_no_tokens = r'([^\(]+?)\s*\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches = re.findall(pattern_no_tokens, model_output)

    if matches:
        # Can handle multiple detections
        results = []
        for match in matches:
            object_name = match[0].strip()
            x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            results.append({
                'object': object_name,
                'bbox': [x1, y1, x2, y2]
            })
        return results[0] if len(results) == 1 else results

    return None  # No valid bbox found
