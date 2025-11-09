"""
Dataset preparation functions for Qwen2-VL fine-tuning.

This module contains functions for DATASET PREPARATION - these run ONCE during
dataset.map() to convert raw OpenFoodFacts data into a format suitable for training.

WHEN USED: During initial dataset loading and preparation (before training loop)
RUNS: Once per dataset sample during dataset.map()

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 1584-1715)
NO MODIFICATIONS - exact copies for testing purposes.

===============================================================================
QUICK REFERENCE - Functions in this module:
===============================================================================

1. system_message
   - System prompt used in all conversations

2. convert_to_conversation_format(example)
   - WHEN: During dataset.map() - runs once per sample
   - INPUT: Raw OpenFoodFacts sample with PIL image
   - OUTPUT: Serializable format with 'IMAGE_PLACEHOLDER' string
   - PURPOSE: Create saveable format (PIL images aren't serializable)

3. _has_image(example)
   - WHEN: During dataset.filter() - validates samples
   - PURPOSE: Filter out corrupted/missing images before training

===============================================================================
"""

# System message used in all conversations
system_message = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


def convert_to_conversation_format(example):
    """
    Convert a raw dataset example to Qwen2-VL conversation format.

    QUICK REFERENCE:
    - WHEN: During dataset.map() - runs once per sample
    - INPUT: Raw OpenFoodFacts sample with PIL image
    - OUTPUT: Serializable format with 'IMAGE_PLACEHOLDER' string
    - PURPOSE: Create saveable format (PIL images aren't serializable)

    ============================================================================
    WHEN USED: During dataset.map() - runs ONCE per sample during preparation
    ============================================================================

    INPUT FORMAT:
        Raw OpenFoodFacts sample:
        {
            'image': <PIL.Image>,
            'objects': {
                'bbox': [[y_min, x_min, y_max, x_max], ...],  # Normalized [0,1]
                'category_name': ['nutrition-table', ...]
            },
            ...
        }

    OUTPUT FORMAT:
        Serializable format with IMAGE_PLACEHOLDER:
        {
            'messages': [
                {'role': 'system', 'content': [{'type': 'text', 'text': '...'}]},
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': 'IMAGE_PLACEHOLDER'},  # ← String placeholder!
                    {'type': 'text', 'text': 'Detect the bounding box...'}
                ]},
                {'role': 'assistant', 'content': [
                    {'type': 'text', 'text': '<|object_ref_start|>nutrition-table<|object_ref_end|>...'}
                ]}
            ],
            'image': <PIL.Image>  # ← Actual image stored separately at top level
        }

    PURPOSE:
        1. Create a format that can be SAVED TO DISK (HuggingFace datasets need serializable data)
        2. Use IMAGE_PLACEHOLDER because PIL images aren't serializable
        3. Store actual PIL image separately to avoid duplication
        4. The placeholder will be replaced with actual image during training (in collator)

    COORDINATE CONVERSION:
        - Input: OpenFoodFacts [y_min, x_min, y_max, x_max] in [0,1]
        - Output: Qwen2-VL (x1,y1),(x2,y2) in [0,1000)

    WHY IMAGE_PLACEHOLDER?
        - HuggingFace dataset.map() needs serializable data (PIL images aren't)
        - The placeholder is replaced with the actual image during training (in collate_fn)
        - Image is stored separately at example['image'] to avoid duplication

    Args:
        example: Dataset sample with 'image' and 'objects' fields

    Returns:
        Dict with 'messages' (conversation with placeholder) and 'image' (PIL object)
    """
    # Validate input
    if 'objects' not in example or 'bbox' not in example['objects']:
        raise ValueError("Missing objects or bbox in example")

    # Extract nutrition table bounding boxes
    bboxes = example['objects']['bbox']
    categories = example['objects']['category_name']  # Fixed: 'category_name' not 'category'

    # Format the assistant response with Qwen2-VL special tokens
    # Convert normalized [0,1] bbox to Qwen's [0,1000) format
    assistant_responses = []
    for bbox, category in zip(bboxes, categories):
        # Validate bbox values are in [0,1] range (with small tolerance for rounding)
        if not all(-0.001 <= coord <= 1.001 for coord in bbox):
            print(f"Warning: bbox coordinates out of [0,1] range: {bbox}")

        # CRITICAL: OpenFoodFacts uses [y_min, x_min, y_max, x_max] format
        # But Qwen2VL expects (x_top_left, y_top_left), (x_bottom_right, y_bottom_right)
        y_min, x_min, y_max, x_max = bbox  # Unpack OpenFoodFacts format

        # Convert to Qwen format: (x,y) coordinates in [0,1000) range
        # Note: multiply by 1000 to convert from [0,1] to [0,1000)
        x1 = int(x_min * 1000)  # x_top_left
        y1 = int(y_min * 1000)  # y_top_left
        x2 = int(x_max * 1000)  # x_bottom_right
        y2 = int(y_max * 1000)  # y_bottom_right

        # Format: <|object_ref_start|>object<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        response = f"<|object_ref_start|>{category}<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
        assistant_responses.append(response)

    # Combine multiple detections if present
    assistant_text = " ".join(assistant_responses)

    # Create conversation format WITHOUT the PIL image embedded
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
                    "image": "IMAGE_PLACEHOLDER"  # Use placeholder instead of actual image
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

    # Return messages and image separately
    return {
        "messages": conversation,
        "image": example['image']  # Store PIL image at top level
    }


def _has_image(example):
    """
    Filter out samples with missing or invalid images.

    QUICK REFERENCE:
    - WHEN: During dataset.filter() - validates samples
    - INPUT: Dataset sample with 'image' field
    - OUTPUT: bool (True if valid, False to filter out)
    - PURPOSE: Filter out corrupted/missing images before training

    ============================================================================
    WHEN USED: During dataset.filter() - runs ONCE per sample to validate
    ============================================================================

    INPUT FORMAT:
        Dataset sample with 'image' field

    OUTPUT:
        bool: True if valid image, False to filter out

    PURPOSE:
        This function is CRUCIAL for preventing None values from leaking into
        process_vision_info during collation/training, which would cause crashes.

    WHY THIS IS NECESSARY:
        - Some dataset samples may have corrupted or missing images
        - PIL Image loading can fail silently, leaving None values
        - The collate_fn and process_vision_info expect valid PIL images
        - Filtering ensures training stability and prevents runtime errors

    VALIDATION:
        - Checks if image is not None
        - Checks if image has 'size' attribute (typical of PIL images)
        - This catches both missing images and corrupted image objects

    Args:
        example: Dataset sample that should contain an 'image' field

    Returns:
        bool: True if example has a valid PIL image with 'size' attribute
    """
    img = example.get('image')
    try:
        # Treat as valid only if it looks like a PIL image
        return (img is not None) and hasattr(img, 'size')
    except Exception:
        return img is not None
