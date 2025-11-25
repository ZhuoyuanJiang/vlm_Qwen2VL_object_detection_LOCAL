"""
Data collation functions and collator classes for Qwen2-VL training.

This module contains functions for BATCH COLLATION - these run during the training
loop for EVERY batch to prepare data for the model.

WHEN USED: Inside collate_fn during training loop
RUNS: Every batch during training (thousands of times)

Extracted from: fine_tuning_vlm_for_object_detection_trl.py

===============================================================================
QUICK REFERENCE - Classes and Functions in this module:
===============================================================================

1. restore_images_in_conversations(messages_list, images_list)
   - WHEN: Inside collate_fn - runs every batch during training
   - INPUT: Messages with 'IMAGE_PLACEHOLDER' + list of PIL images
   - OUTPUT: Messages with actual PIL images restored

2. collate_fn_fixed_1 (class)
   - Basic collator: masks only pad + vision tokens
   - Trains on: system + user + assistant messages

3. collate_fn_fixed_fixed1 (class)
   - Same as fixed_1 + optional Flash Attention monkey patch
   - Use use_flash_patch=True to enable Flash Attention dtype fix

4. collate_fn_fixed_3 (class)
   - Assistant-only collator: masks everything except assistant response
   - Trains on: only assistant response tokens (~30% of sequence)

===============================================================================
"""

import torch
from qwen_vl_utils import process_vision_info


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
                    if text_value is not None and text_value != 'None':
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
                            'image': image
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


# =============================================================================
# COLLATOR V1: Basic - masks only pad + vision tokens
# =============================================================================

class collate_fn_fixed_1:
    """
    Basic collate function that masks pad and vision tokens.

    This collator:
    1. Restores IMAGE_PLACEHOLDER with actual PIL images
    2. Uses process_vision_info to properly extract and format images
    3. Applies chat template to get formatted text
    4. Tokenizes text and processes images together
    5. Creates labels with masking for pad and vision tokens

    Tokens masked (-100):
    - pad_token_id
    - <|vision_start|>
    - <|vision_end|>
    - <|image_pad|>

    Args:
        processor: Qwen2VLProcessor instance
        model: Model instance (not used directly, kept for API compatibility)
    """

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
        self.masked_token_ids = [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),
            self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),
            self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        ]

    def __call__(self, batch):
        # Extract messages and images from each sample
        messages_list = [sample['messages'] for sample in batch]
        images_list = [sample.get('image', None) for sample in batch]

        # Restore images in conversations
        all_conversations = restore_images_in_conversations(messages_list, images_list)

        # Apply chat template to get text
        text = self.processor.apply_chat_template(
            all_conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process vision info to get images
        image, video = process_vision_info(all_conversations)

        # Process texts and images together
        batch_inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )

        # Fix dtype for input_ids
        if 'input_ids' in batch_inputs:
            batch_inputs['input_ids'] = batch_inputs['input_ids'].to(torch.long)

        # Create labels from input_ids
        labels = batch_inputs["input_ids"].clone()
        for token_id in self.masked_token_ids:
            labels[labels == token_id] = -100
        batch_inputs["labels"] = labels

        return batch_inputs


# =============================================================================
# COLLATOR V2: With optional Flash Attention patch
# =============================================================================

class collate_fn_fixed_fixed1:
    """
    Collator with optional Flash Attention dtype fix.

    The Flash Attention cu_seqlens dtype error occurs because Flash Attention
    expects int32 tensors but they're created as int64. This collator can
    optionally apply a monkey patch to fix this.

    Args:
        processor: Qwen2VLProcessor instance
        model: Model instance (not used directly, kept for API compatibility)
        use_flash_patch: If True, apply Flash Attention monkey patch (default: False)
    """

    def __init__(self, processor, model, use_flash_patch=False):
        self.processor = processor
        self.model = model
        self.masked_token_ids = [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),
            self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),
            self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        ]
        self.call_count = 0

        # Only apply Flash Attention patch if requested
        if use_flash_patch:
            self._patch_flash_attention()

    def _patch_flash_attention(self):
        """Monkey-patch Flash Attention to fix cu_seqlens dtype."""
        import flash_attn.flash_attn_interface as flash_interface

        # Store original _flash_attn_varlen_forward function
        original_flash_varlen_forward = flash_interface._flash_attn_varlen_forward

        def patched_flash_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                        max_seqlen_q, max_seqlen_k, *args, **kwargs):
            # Convert cu_seqlens to int32 if needed
            if cu_seqlens_q is not None and cu_seqlens_q.dtype != torch.int32:
                cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            if cu_seqlens_k is not None and cu_seqlens_k.dtype != torch.int32:
                cu_seqlens_k = cu_seqlens_k.to(torch.int32)

            return original_flash_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                                max_seqlen_q, max_seqlen_k, *args, **kwargs)

        flash_interface._flash_attn_varlen_forward = patched_flash_varlen_forward

        # Also patch the FlashAttnVarlenFunc.forward method
        from flash_attn.flash_attn_interface import FlashAttnVarlenFunc
        original_forward = FlashAttnVarlenFunc.forward

        @staticmethod
        def patched_forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, *args):
            if cu_seqlens_q is not None and cu_seqlens_q.dtype != torch.int32:
                cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            if cu_seqlens_k is not None and cu_seqlens_k.dtype != torch.int32:
                cu_seqlens_k = cu_seqlens_k.to(torch.int32)

            return original_forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k, *args)

        FlashAttnVarlenFunc.forward = patched_forward
        print("✅ Applied Flash Attention cu_seqlens dtype patch")

    def __call__(self, batch):
        # Extract messages and images from each sample
        messages_list = [sample['messages'] for sample in batch]
        images_list = [sample.get('image', None) for sample in batch]

        # Restore images in conversations
        all_conversations = restore_images_in_conversations(messages_list, images_list)

        # Apply chat template
        text = self.processor.apply_chat_template(
            all_conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process vision info
        image, video = process_vision_info(all_conversations)

        # Process texts and images together
        batch_inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )

        # Ensure correct dtypes
        if 'input_ids' in batch_inputs:
            batch_inputs['input_ids'] = batch_inputs['input_ids'].to(torch.long)

        if 'attention_mask' in batch_inputs:
            batch_inputs['attention_mask'] = batch_inputs['attention_mask'].to(torch.long)

        # Create labels with masking
        labels = batch_inputs["input_ids"].clone()
        for token_id in self.masked_token_ids:
            labels[labels == token_id] = -100
        batch_inputs["labels"] = labels

        return batch_inputs


# =============================================================================
# COLLATOR V3: Assistant-only training
# =============================================================================

class collate_fn_fixed_3:
    """
    Collator that masks everything except assistant responses.

    This ensures the model only learns the actual detection task (bbox coordinates)
    rather than memorizing static prompts.

    Tokens masked (-100):
    - All padding tokens
    - All vision tokens
    - All system message tokens
    - All user message tokens

    Tokens trained on:
    - Only assistant response tokens (~30% of sequence)

    Args:
        processor: Qwen2VLProcessor instance
        model: Model instance (not used directly, kept for API compatibility)
    """

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

        # Track special tokens to mask
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.vision_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),
            self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),
            self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        ]

        # Assistant markers for finding response boundaries
        self.assistant_start_token = "<|im_start|>assistant\n"
        self.assistant_end_token = "<|im_end|>"

        # For debugging
        self.debug = False
        self.debug_count = 0
        self.max_debug = 3

    def __call__(self, batch):
        # Step 1: Extract and prepare messages with actual images
        messages_list = [sample['messages'] for sample in batch]
        images_list = [sample.get('image', None) for sample in batch]

        # Restore images in conversations
        all_conversations = restore_images_in_conversations(messages_list, images_list)

        # Step 2: Apply chat template
        texts = self.processor.apply_chat_template(
            all_conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        # Step 3: Process vision info
        images, videos = process_vision_info(all_conversations)

        # Step 4: Tokenize and process
        batch_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )

        # Step 5: Create labels - start with everything masked
        labels = batch_inputs["input_ids"].clone()
        labels[:, :] = -100  # Mask everything initially

        # Step 6: Find and unmask ONLY assistant responses
        for batch_idx, text in enumerate(texts):
            input_ids = batch_inputs["input_ids"][batch_idx]

            # Find assistant response boundaries in the text
            assistant_start_pos = text.find(self.assistant_start_token)
            if assistant_start_pos == -1:
                # Try without newline
                assistant_start_pos = text.find("<|im_start|>assistant")

            if assistant_start_pos != -1:
                # Find where actual response starts (after the marker)
                response_start_in_text = assistant_start_pos + len(self.assistant_start_token)

                # Find end of assistant response
                assistant_end_pos = text.find(self.assistant_end_token, response_start_in_text)

                if assistant_end_pos != -1:
                    # Extract just the response text
                    response_text = text[response_start_in_text:assistant_end_pos]

                    # Tokenize the response to find it in input_ids
                    response_tokens = self.processor.tokenizer.encode(
                        response_text,
                        add_special_tokens=False
                    )

                    if response_tokens:
                        # Find this sequence in the full input_ids
                        for i in range(len(input_ids) - len(response_tokens) + 1):
                            if input_ids[i:i+len(response_tokens)].tolist() == response_tokens:
                                # Unmask these tokens for training
                                labels[batch_idx, i:i+len(response_tokens)] = input_ids[i:i+len(response_tokens)]
                                break

        # Step 7: Debug output (only for first few samples)
        if self.debug and self.debug_count < self.max_debug:
            self.debug_count += 1
            non_masked = (labels != -100).sum().item()
            total = labels.numel()
            if non_masked > 0:
                sample_labels = labels[0]
                trained_indices = (sample_labels != -100).nonzero(as_tuple=True)[0]
                if len(trained_indices) > 0:
                    trained_tokens = batch_inputs["input_ids"][0][trained_indices]
                    decoded = self.processor.tokenizer.decode(trained_tokens, skip_special_tokens=False)
                    print(f"[collate_fn_fixed_3] Sample {self.debug_count}: Training on {len(trained_indices)} tokens")
                    print(f"  Content: '{decoded}'")
                    print(f"  This is {100*len(trained_indices)/len(sample_labels):.1f}% of {len(sample_labels)} total tokens")

        batch_inputs["labels"] = labels
        return batch_inputs
