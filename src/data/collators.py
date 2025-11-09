"""
Data collation functions and collator classes for Qwen2-VL training.

This module contains functions for BATCH COLLATION - these run during the training
loop for EVERY batch to prepare data for the model.

WHEN USED: Inside collate_fn during training loop
RUNS: Every batch during training (thousands of times)

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 2506-2562)
NO MODIFICATIONS - exact copies for testing purposes.

===============================================================================
QUICK REFERENCE - Functions in this module:
===============================================================================

1. restore_images_in_conversations(messages_list, images_list)
   - WHEN: Inside collate_fn - runs every batch during training
   - INPUT: Messages with 'IMAGE_PLACEHOLDER' + list of PIL images
   - OUTPUT: Messages with actual PIL images restored
   - PURPOSE: Replace placeholder with real images for apply_chat_template
   - CRITICAL: Output must be exact format for Qwen2-VL!

2. TODO: Collator classes (to be added)
   - Qwen2VLCollatorV1 (basic implementation)
   - Qwen2VLCollatorV2 (improved with better masking)
   - Qwen2VLCollatorV3 (final optimized version)

===============================================================================
"""


def restore_images_in_conversations(messages_list, images_list):
    """
    Replace IMAGE_PLACEHOLDER with actual PIL images and clean up None values.

    QUICK REFERENCE:
    - WHEN: Inside collate_fn - runs every batch during training
    - INPUT: Messages with 'IMAGE_PLACEHOLDER' + list of PIL images
    - OUTPUT: Messages with actual PIL images restored
    - PURPOSE: Replace placeholder with real images for apply_chat_template
    - CRITICAL: Output must be exact format for Qwen2-VL!

    ============================================================================
    WHEN USED: Inside collate_fn - runs EVERY BATCH during training
    ============================================================================

    This is the CRITICAL step that happens BEFORE apply_chat_template.
    The output format must be EXACTLY correct for Qwen2-VL to work!

    INPUT FORMAT:
        messages_list: List of conversations with IMAGE_PLACEHOLDER
        [
            [  # Conversation 1 (from convert_to_conversation_format)
                {'role': 'system', 'content': [{'type': 'text', 'text': '...'}]},
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': 'IMAGE_PLACEHOLDER'},  # ← String placeholder
                    {'type': 'text', 'text': '...'}
                ]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '...'}]}
            ],
            # More conversations...
        ]

        images_list: List of actual PIL Image objects
        [<PIL.Image>, <PIL.Image>, ...]

    OUTPUT FORMAT:
        all_conversations: List of conversations ready for apply_chat_template
        [
            [  # Conversation 1
                {'role': 'system', 'content': [{'type': 'text', 'text': '...'}]},
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': <PIL.Image>},  # ← Actual PIL image restored!
                    {'type': 'text', 'text': '...'}
                ]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '...'}]}
            ],
            # More conversations...
        ]

    PURPOSE:
        1. Restore actual PIL images (replace 'IMAGE_PLACEHOLDER' string with <PIL.Image>)
        2. Clean up any None values that might exist in content
        3. Filter out empty content items
        4. Prepare format that Qwen2-VL's apply_chat_template and process_vision_info expect

    CRITICAL REQUIREMENTS FOR OUTPUT:
        ✅ Image must be actual PIL.Image object (NOT 'IMAGE_PLACEHOLDER' string)
        ✅ No None values in content lists
        ✅ No string 'None' in text content
        ✅ Proper structure: list of conversations → list of messages → dict with role/content
        ✅ Each message must have non-empty content list

    WHY THIS STEP EXISTS:
        - dataset.map() stores IMAGE_PLACEHOLDER (can't serialize PIL images)
        - During training, we need actual PIL images for the model
        - This function bridges the gap: placeholder → actual image
        - Happens every batch because we can't store PIL images in dataset

    FLOW IN TRAINING:
        1. DataLoader loads batch → samples have IMAGE_PLACEHOLDER
        2. collate_fn calls this function → replaces with actual PIL images
        3. apply_chat_template → converts to text with special tokens
        4. process_vision_info → extracts images for vision encoder
        5. processor → tokenizes and creates pixel values

    Args:
        messages_list: List of message conversations (each with IMAGE_PLACEHOLDER)
        images_list: List of actual PIL Image objects

    Returns:
        all_conversations: List of conversations ready for apply_chat_template

    Raises:
        ValueError: If batch contains no valid images
    """
    # Filter out samples without images
    valid_pairs = [(m, img) for m, img in zip(messages_list, images_list) if img is not None]
    if not valid_pairs:
        raise ValueError("Batch contains no valid images.")

    messages_list, images_list = zip(*valid_pairs)
    messages_list = list(messages_list)
    images_list = list(images_list)


    all_conversations = []  # This will hold complete conversations, each conversation is a list of 3 messages (system, user, assistant)

    for messages, image in zip(messages_list, images_list):
        messages_with_image = []
        # Clean up messages: remove None values and restore images
        for msg in messages:
            msg_copy = {'role': msg['role'], 'content': []}

            for content_item in msg['content']:
                # Skip None entries entirely
                if content_item is None:
                    continue

                # Process text content - filter out None text values
                if content_item.get('type') == 'text':
                    text_value = content_item.get('text')
                    if text_value is not None and text_value != 'None':  # Check for actual None and string 'None'
                        msg_copy['content'].append({
                            'type': 'text',
                            'text': text_value
                        })
                # Process image content - replace IMAGE_PLACEHOLDER with actual PIL image
                elif content_item.get('type') == 'image' and msg['role'] == 'user':
                    # Check if it's IMAGE_PLACEHOLDER and replace with actual image
                    image_value = content_item.get('image')
                    if image_value == 'IMAGE_PLACEHOLDER' or image_value is None:
                        # Replace with the actual PIL image from top level
                        msg_copy['content'].append({
                            'type': 'image',
                            'image': image  # Use the actual PIL image
                        })
                    elif image_value and image_value != 'None':
                        # Use existing image if it's not placeholder
                        msg_copy['content'].append({
                            'type': 'image',
                            'image': image_value
                        })

            if msg_copy['content']:
                messages_with_image.append(msg_copy)

        # Add the complete conversation (3 messages) to collection
        all_conversations.append(messages_with_image)

    return all_conversations


# TODO: Add collator classes here
# - Qwen2VLCollatorV1 (basic implementation)
# - Qwen2VLCollatorV2 (improved with better masking)
# - Qwen2VLCollatorV3 (final optimized version)
# These will be extracted in the next step
