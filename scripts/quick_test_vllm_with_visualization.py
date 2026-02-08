"""
Manual vLLM smoke test with local bbox visualization.

Purpose:
    - Send one multimodal request to vLLM and inspect the returned text output.
    - Parse bbox tokens and draw a local visualization for quick human checking.

Relation to quick_test_vllm_api.py:
    - Both scripts make the same vLLM API call.
    - quick_test_vllm_api.py prints raw output only.
    - This script adds local post-processing (parse + draw + save image).

When to use:
    - Manual sanity checks and demos.
    - Fast debugging of output format and bbox plausibility.

When not to use:
    - Not a formal benchmark script.
    - Not a batch accuracy evaluation pipeline.

Usage:
    python scripts/quick_test_vllm_with_visualization.py

Prerequisite:
    A vLLM server is already running at http://localhost:8000
    with served model name "qwen2vl-nutrition".
"""

import base64
import re
import requests
from datasets import load_dataset
from PIL import Image, ImageDraw

# =============================================================================
# Configuration
# =============================================================================
VLLM_HOST = "localhost"  # Change to "vllab8" if testing from another server
VLLM_PORT = 8000
MODEL_NAME = "qwen2vl-nutrition"

SYSTEM_PROMPT = """You are a nutrition label detector. Your task is to identify nutrition tables/panels in food product images.

When you detect a nutrition table, output its location using this exact format:
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates are in [0,1000) range where (0,0) is top-left.
If no nutrition table is found, say "No nutrition table detected."
"""

USER_PROMPT = "Detect the bounding box coordinates for the nutrition facts table in this image."


# =============================================================================
# Parsing function (from fine_tuning_vlm_for_object_detection_trl.py)
# =============================================================================
def parse_qwen_bbox_output(model_output):
    """
    Parse Qwen2-VL model output to extract bounding box coordinates.

    Args:
        model_output: String output from the model

    Returns:
        Dict with 'object' name and 'bbox' coordinates, or None if parsing fails
    """
    # Remove <|im_end|> token if present
    model_output = model_output.replace('<|im_end|>', '').strip()

    # Pattern 1: WITH special tokens (skip_special_tokens=False)
    # Format: <|object_ref_start|>object name<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    pattern_with_tokens = r'<\|object_ref_start\|>(.+?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches = re.findall(pattern_with_tokens, model_output)

    if matches:
        results = []
        for match in matches:
            object_name = match[0]
            x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            results.append({
                'object': object_name,
                'bbox': [x1, y1, x2, y2]
            })
        return results[0] if len(results) == 1 else results

    # Pattern 2: WITHOUT special tokens (skip_special_tokens=True)
    # Format: "object name(x1,y1),(x2,y2)"
    pattern_no_tokens = r'([^\(]+?)\s*\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches = re.findall(pattern_no_tokens, model_output)

    if matches:
        results = []
        for match in matches:
            object_name = match[0].strip()
            x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            results.append({
                'object': object_name,
                'bbox': [x1, y1, x2, y2]
            })
        return results[0] if len(results) == 1 else results

    return None


def visualize_bbox_on_image(image, bbox_data, output_path=None):
    """
    Visualize bounding boxes on an image.

    Args:
        image: PIL Image object
        bbox_data: Dict with 'object' and 'bbox' keys
        output_path: Optional path to save the image

    Returns:
        PIL Image with bounding boxes drawn
    """
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    width, height = img_with_bbox.size

    if bbox_data is None:
        print("No bounding box data to visualize")
        return img_with_bbox

    # Handle single or multiple bboxes
    if isinstance(bbox_data, dict):
        bbox_data = [bbox_data]

    colors = ['red', 'blue', 'green', 'yellow']

    for idx, data in enumerate(bbox_data):
        if data is None:
            continue

        color = colors[idx % len(colors)]
        bbox = data['bbox']
        object_name = data.get('object', 'unknown')

        # Convert from Qwen's [0,1000] to pixel coordinates
        x1 = int(bbox[0] * width / 1000)
        y1 = int(bbox[1] * height / 1000)
        x2 = int(bbox[2] * width / 1000)
        y2 = int(bbox[3] * height / 1000)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"{object_name}"
        draw.text((x1, y1 - 15), label, fill=color)

    if output_path:
        img_with_bbox.save(output_path)
        print(f"Saved visualization to: {output_path}")

    return img_with_bbox


def test_vllm_inference(image_path: str):
    """
    Test vLLM inference with an image.

    Args:
        image_path: Path to the image file

    Returns:
        Parsed bbox data or None
    """
    # Encode image to base64
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Send request to vLLM
    response = requests.post(
        f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": USER_PROMPT}
                ]}
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "skip_special_tokens": False
        },
        timeout=60
    )

    result = response.json()
    model_output = result['choices'][0]['message']['content']

    return model_output


# =============================================================================
# Main test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Deployment Test - Qwen2-VL Nutrition Detection")
    print("=" * 60)

    # Load test image from dataset
    print("\n1. Loading test image from HuggingFace dataset...")
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")
    test_image = ds[0]['image']
    test_image.save('/tmp/test_nutrition.jpg')
    print(f"   Image size: {test_image.size}")
    print("   Saved to: /tmp/test_nutrition.jpg")

    # Get ground truth if available
    if 'objects' in ds[0]:
        gt_bbox = ds[0]['objects']['bbox'][0] if ds[0]['objects']['bbox'] else None
        print(f"   Ground truth bbox: {gt_bbox}")

    # Test vLLM inference
    print(f"\n2. Sending request to vLLM server at {VLLM_HOST}:{VLLM_PORT}...")
    model_output = test_vllm_inference('/tmp/test_nutrition.jpg')

    print("\n" + "=" * 60)
    print("Model Output (raw):")
    print("=" * 60)
    print(model_output)

    # Parse the output
    print("\n" + "=" * 60)
    print("Parsed Result:")
    print("=" * 60)
    parsed = parse_qwen_bbox_output(model_output)

    if parsed:
        print(f"   Object: {parsed['object']}")
        print(f"   Bbox (normalized 0-1000): {parsed['bbox']}")

        # Convert to pixel coordinates for display
        w, h = test_image.size
        x1 = int(parsed['bbox'][0] * w / 1000)
        y1 = int(parsed['bbox'][1] * h / 1000)
        x2 = int(parsed['bbox'][2] * w / 1000)
        y2 = int(parsed['bbox'][3] * h / 1000)
        print(f"   Bbox (pixel coords): [{x1}, {y1}, {x2}, {y2}]")

        # Visualize
        print("\n3. Visualizing bounding box...")
        vis_image = visualize_bbox_on_image(test_image, parsed, '/tmp/test_nutrition_result.jpg')
        print("   Saved visualization to: /tmp/test_nutrition_result.jpg")
    else:
        print("   Failed to parse bounding box from model output")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
