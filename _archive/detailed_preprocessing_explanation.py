#!/usr/bin/env python3
"""
Detailed explanation and full examples of the data preprocessing function
"""

from PIL import Image
import json

# System message (keeping as is per user request)
system_message = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def convert_to_conversation_format(example):
    """
    Convert a dataset example to Qwen2-VL conversation format.
    
    THIS FUNCTION MAPS EACH SAMPLE TO A LIST OF 3 DICTS (ONE FOR EACH ROLE):
    1. System role dict - contains system instructions
    2. User role dict - contains image and question
    3. Assistant role dict - contains the expected answer
    
    The list is stored under the 'messages' key.
    """
    # Extract nutrition table bounding boxes
    bboxes = example['objects']['bbox']
    categories = example['objects']['category_name']
    
    # Format the assistant response with Qwen2-VL special tokens
    assistant_responses = []
    for bbox, category in zip(bboxes, categories):
        y_min, x_min, y_max, x_max = bbox
        x1 = int(x_min * 1000)
        y1 = int(y_min * 1000)
        x2 = int(x_max * 1000)
        y2 = int(y_max * 1000)
        response = f"<|object_ref_start|>{category}<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
        assistant_responses.append(response)
    
    assistant_text = " ".join(assistant_responses)
    
    # HERE'S THE LIST OF 3 DICTS (ONE FOR EACH ROLE):
    conversation = [
        # DICT 1: SYSTEM ROLE
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        },
        # DICT 2: USER ROLE
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "IMAGE_PLACEHOLDER"  # Explained below
                },
                {
                    "type": "text",
                    "text": "Detect the bounding box of the nutrition table."
                }
            ]
        },
        # DICT 3: ASSISTANT ROLE
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
    
    return {
        "messages": conversation,  # The list of 3 dicts
        "image": example['image']  # PIL image stored separately
    }

print("="*80)
print("UNDERSTANDING THE FUNCTION: How it maps to 3 dicts")
print("="*80)

# Create mock examples
dummy_image = Image.new('RGB', (2592, 1944), color='white')

# Example 1: Single nutrition table
example1 = {
    'image': dummy_image,  # This would be a real PIL.JpegImagePlugin.JpegImageFile in actual data
    'objects': {
        'bbox': [[0.057, 0.014, 0.603, 0.991]],  # [y_min, x_min, y_max, x_max]
        'category_name': ['nutrition-table']
    }
}

# Example 2: Multiple nutrition tables
example2 = {
    'image': dummy_image,
    'objects': {
        'bbox': [
            [0.100, 0.200, 0.400, 0.500],
            [0.600, 0.700, 0.800, 0.900]
        ],
        'category_name': ['nutrition-table', 'nutrition-table']
    }
}

# Process examples
formatted1 = convert_to_conversation_format(example1)
formatted2 = convert_to_conversation_format(example2)

print("\n" + "="*80)
print("FULL EXAMPLE 1: Single nutrition table (NO TRUNCATION)")
print("="*80)
print("\nformatted1 = {")
print(f"  'messages': [")
for i, msg in enumerate(formatted1['messages']):
    print(f"    {{ # Dict {i+1} of 3")
    print(f"      'role': '{msg['role']}',")
    print(f"      'content': [")
    for content in msg['content']:
        print(f"        {{")
        print(f"          'type': '{content['type']}',")
        if content['type'] == 'text':
            # Print FULL text without truncation
            print(f"          'text': '''{content['text']}'''")
        else:
            print(f"          'image': '{content['image']}'")
        print(f"        }},")
    print(f"      ]")
    print(f"    }},")
print(f"  ],")
print(f"  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size={formatted1['image'].size} at {hex(id(formatted1['image']))}>")
print("}")

print("\n" + "="*80)
print("FULL EXAMPLE 2: Multiple nutrition tables (NO TRUNCATION)")
print("="*80)
print("\nformatted2 = {")
print(f"  'messages': [")
for i, msg in enumerate(formatted2['messages']):
    print(f"    {{ # Dict {i+1} of 3")
    print(f"      'role': '{msg['role']}',")
    print(f"      'content': [")
    for content in msg['content']:
        print(f"        {{")
        print(f"          'type': '{content['type']}',")
        if content['type'] == 'text':
            # Print FULL text without truncation
            print(f"          'text': '''{content['text']}'''")
        else:
            print(f"          'image': '{content['image']}'")
        print(f"        }},")
    print(f"      ]")
    print(f"    }},")
print(f"  ],")
print(f"  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size={formatted2['image'].size} at {hex(id(formatted2['image']))}>")
print("}")

print("\n" + "="*80)
print("WHY IMAGE_PLACEHOLDER INSTEAD OF THE ACTUAL IMAGE?")
print("="*80)
print("""
The IMAGE_PLACEHOLDER is NECESSARY and CORRECT. Here's why:

1. DATASET.MAP() SERIALIZATION ISSUE:
   - When you call dataset.map(convert_to_conversation_format), HuggingFace needs to 
     serialize the results to save them
   - PIL images CANNOT be serialized within nested dicts/lists
   - If we put the actual PIL image in messages[1]['content'][0]['image'], 
     dataset.map() would crash with a serialization error

2. TWO-PART STORAGE SOLUTION:
   - The actual PIL image is stored at the TOP LEVEL: formatted_sample['image']
   - The placeholder marks WHERE the image should go in the conversation
   - This allows dataset.map() to work properly

3. COLLATOR REPLACEMENT:
   - During training, the collate_fn replaces IMAGE_PLACEHOLDER with the actual image
   - See this code in collate_fn:
     ```python
     if content_item.get('image') == 'IMAGE_PLACEHOLDER':
         if images_list[i] is not None:
             content_item['image'] = images_list[i]  # Replace with actual image
     ```

4. THIS IS THE STANDARD APPROACH:
   - Many VLM training pipelines use this pattern
   - It's not overcomplicated - it's solving a real technical constraint
""")

print("\n" + "="*80)
print("WHY THE _has_image FILTER?")
print("="*80)
print("""
The _has_image filter is IMPORTANT for robustness:

```python
def _has_image(example):
    img = example.get('image')
    try:
        # Treat as valid only if it looks like a PIL image
        return (img is not None) and hasattr(img, 'size')
    except Exception:
        return img is not None
```

REASONS FOR THIS FILTER:

1. DATASET QUALITY ISSUES:
   - Some samples in the dataset might have None or corrupted images
   - This can happen due to download failures or dataset creation issues

2. PREVENTS TRAINING CRASHES:
   - Without filtering, when collate_fn calls process_vision_info(messages_with_image),
     it would crash on None images
   - The error would be: "AttributeError: 'NoneType' object has no attribute 'size'"

3. ROBUST CHECKING:
   - Checks both that image is not None AND has 'size' attribute (PIL images have .size)
   - The try/except handles edge cases where checking attributes might fail

4. CLEAN TRAINING DATA:
   - Ensures only valid samples with proper images reach the training loop
   - Better to filter early than crash during training

This is GOOD DEFENSIVE PROGRAMMING, not overcomplication.
""")

print("\n" + "="*80)
print("IS THE CODE OVERCOMPLICATED? CAN WE SIMPLIFY?")
print("="*80)
print("""
The code is NOT overcomplicated. Each part serves a specific purpose:

✅ NECESSARY COMPLEXITY:
1. IMAGE_PLACEHOLDER - Required for dataset serialization
2. Coordinate conversion - Required to match Qwen2-VL format
3. Special tokens - Required for Qwen2-VL to understand the task
4. _has_image filter - Prevents training crashes
5. Separate image storage - Enables proper batching

❌ WHAT WE CANNOT SIMPLIFY:
- Cannot put PIL image directly in messages (serialization fails)
- Cannot skip coordinate conversion (wrong format for model)
- Cannot remove special tokens (model won't understand output)
- Cannot skip filtering (training would crash on bad samples)

✅ WHAT WE COULD POTENTIALLY SIMPLIFY (but shouldn't):
- Could hardcode assumptions about data (but less flexible)
- Could skip error handling (but less robust)
- Could merge some steps (but less readable)

CONCLUSION: The current implementation is appropriately complex for the task.
Each piece of "complexity" solves a real technical requirement.
""")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
1. ✅ The function DOES map each sample to 3 dicts (system, user, assistant roles)
2. ✅ IMAGE_PLACEHOLDER is necessary for dataset serialization
3. ✅ The actual image IS preserved at formatted_sample['image']
4. ✅ The _has_image filter prevents training crashes
5. ✅ The code complexity is justified by technical requirements

The implementation is correct and follows best practices for VLM training.
""")