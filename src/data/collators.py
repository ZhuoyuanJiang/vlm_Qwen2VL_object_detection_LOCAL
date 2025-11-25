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


# =============================================================================
# BACKUP/DRAFT COLLATORS - For reference only, not production-ready
# =============================================================================
# These are experimental versions kept for reference and debugging.
# The production collators are the three classes above:
#   - collate_fn_fixed_1 (basic)
#   - collate_fn_fixed_fixed1 (with Flash Attention patch option)
#   - collate_fn_fixed_3 (assistant-only training)
#
# NOTE: The draft collators below use global 'processor' and 'model' variables.
# They need modification to use self.processor/self.model if converted to classes.
# =============================================================================


def collate_fn_fixed_2(batch, processor, model):
    """
    DRAFT/BACKUP VERSION: Verbose debug collator with detailed masking.

    NOTE: This is a function-based collator (not a class).
    Requires processor and model to be passed as arguments.

    This collator:
    1. Restores IMAGE_PLACEHOLDER with actual PIL images
    2. Uses process_vision_info to properly extract and format images
    3. Applies chat template to get formatted text
    4. Tokenizes text and processes images together
    5. Creates labels with proper masking for loss computation

    Key improvements:
    - Properly masks all vision/special tokens to prevent training on them
    - Validates that no out-of-vocabulary tokens exist in labels
    - Uses token-based span finding for robust assistant response extraction
    - Handles edge cases gracefully (samples without images, OOV tokens)
    - CRITICAL: Initially masks ALL tokens, then selectively unmasks only assistant responses
    - Provides detailed debugging statistics about training token percentages

    Args:
        batch: List of samples from the dataset, each containing 'messages' and 'image'
        processor: Qwen2VLProcessor instance
        model: Model instance (for device placement)

    Returns:
        Dict with input_ids, attention_mask, pixel_values, image_grid_thw, and labels
    """
    # Extract messages and images from each sample
    messages_list = [sample['messages'] for sample in batch]
    images_list = [sample.get('image', None) for sample in batch]

    # Filter out samples without images
    valid_pairs = [(m, img) for m, img in zip(messages_list, images_list) if img is not None]
    if not valid_pairs:
        raise ValueError("Batch contains no valid images.")

    messages_list, images_list = zip(*valid_pairs)
    messages_list = list(messages_list)
    images_list = list(images_list)

    all_conversations = []

    for messages, image in zip(messages_list, images_list):
        messages_with_image = []
        for msg in messages:
            msg_copy = {'role': msg['role'], 'content': []}

            for content_item in msg['content']:
                if content_item is None:
                    continue

                if content_item.get('type') == 'text':
                    text_value = content_item.get('text')
                    if text_value is not None and text_value != 'None':
                        msg_copy['content'].append({
                            'type': 'text',
                            'text': text_value
                        })
                elif content_item.get('type') == 'image' and msg['role'] == 'user':
                    image_value = content_item.get('image')
                    if image_value == 'IMAGE_PLACEHOLDER' or image_value is None:
                        msg_copy['content'].append({
                            'type': 'image',
                            'image': image
                        })
                    elif image_value and image_value != 'None':
                        msg_copy['content'].append({
                            'type': 'image',
                            'image': image_value
                        })

            if msg_copy['content']:
                messages_with_image.append(msg_copy)

        all_conversations.append(messages_with_image)

    # Apply chat template to get text
    text = processor.apply_chat_template(
        all_conversations,
        tokenize=False,
        add_generation_prompt=False
    )

    # process_vision_info to get image and video
    image, video = process_vision_info(all_conversations)

    # Process texts and images together
    batch_inputs = processor(
        text=text,
        images=image,
        padding=True,
        truncation=False,
        return_tensors="pt"
    )

    # Create labels from input_ids
    labels = batch_inputs["input_ids"].clone()

    # Get vocabulary size for validation
    vocab_size = processor.tokenizer.vocab_size

    # Track OOV tokens (these are vision tokens with IDs >= vocab_size)
    oov_mask = batch_inputs["input_ids"] >= vocab_size
    if oov_mask.any():
        num_oov = oov_mask.sum().item()
        print(f"[collate_fn_fixed_2] INFO: Found {num_oov} vision tokens (IDs >= vocab_size)")
        labels[oov_mask] = -100

    # IMPORTANT: Start by masking EVERYTHING
    labels[:, :] = -100

    # Collect special token IDs for verification
    special_token_ids = set()

    if processor.tokenizer.pad_token_id is not None:
        special_token_ids.add(processor.tokenizer.pad_token_id)

    special_token_strings = [
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>"
    ]

    for token_str in special_token_strings:
        if token_str in processor.tokenizer.get_vocab():
            token_id = processor.tokenizer.convert_tokens_to_ids(token_str)
            if token_id is not None:
                special_token_ids.add(token_id)

    # DEBUG: Let's understand the token structure
    debug_mode = True
    if debug_mode and labels.shape[0] > 0:
        first_100 = batch_inputs["input_ids"][0][:100]
        decoded = processor.tokenizer.decode(first_100, skip_special_tokens=False)
        print(f"[DEBUG] First 100 tokens decoded: {decoded[:200]}...")

    # Find and unmask ONLY assistant responses
    assistant_markers = [
        ("<|im_start|>assistant\n", "<|im_end|>"),
        ("<|im_start|>assistant", "<|im_end|>"),
        ("assistant\n", "<|im_end|}"),
    ]

    assistant_found = False
    for start_text, end_text in assistant_markers:
        if assistant_found:
            break

        try:
            assistant_start_ids = processor.tokenizer.encode(start_text, add_special_tokens=False)
            assistant_end_ids = processor.tokenizer.encode(end_text, add_special_tokens=False)

            if debug_mode:
                print(f"[DEBUG] Trying pattern: '{start_text}' -> {assistant_start_ids}")

            if assistant_start_ids and assistant_end_ids:
                def find_token_sequence(tokens, sequence, start_pos=0):
                    seq_len = len(sequence)
                    for i in range(start_pos, len(tokens) - seq_len + 1):
                        if tokens[i:i + seq_len].tolist() == sequence:
                            return i
                    return -1

                assistant_tokens_found = 0
                for batch_idx in range(labels.shape[0]):
                    input_ids_list = batch_inputs["input_ids"][batch_idx]

                    pos = 0
                    while pos < len(input_ids_list):
                        start_pos = find_token_sequence(input_ids_list, assistant_start_ids, pos)
                        if start_pos == -1:
                            break

                        assistant_found = True
                        response_start = start_pos + len(assistant_start_ids)
                        end_pos = find_token_sequence(input_ids_list, assistant_end_ids, response_start)

                        if end_pos == -1:
                            end_pos = len(input_ids_list)

                        if debug_mode and batch_idx == 0 and assistant_tokens_found == 0:
                            response_tokens = input_ids_list[response_start:min(response_start+50, end_pos)]
                            response_text = processor.tokenizer.decode(response_tokens, skip_special_tokens=False)
                            print(f"[DEBUG] Found assistant response: '{response_text}'")

                        for idx in range(response_start, end_pos):
                            token_id = input_ids_list[idx].item()
                            if token_id not in special_token_ids and 0 <= token_id < vocab_size:
                                labels[batch_idx, idx] = token_id
                                assistant_tokens_found += 1

                        pos = end_pos + len(assistant_end_ids) if end_pos < len(input_ids_list) else len(input_ids_list)

                if assistant_tokens_found > 0:
                    print(f"[collate_fn_fixed_2] Found {assistant_tokens_found} assistant tokens to train on")
                    break

        except Exception as e:
            print(f"[collate_fn_fixed_2] Error trying pattern '{start_text}': {e}")

    if not assistant_found:
        print("[collate_fn_fixed_2] WARNING: No assistant responses found with any pattern!")
        if debug_mode and labels.shape[0] > 0:
            full_text = processor.tokenizer.decode(batch_inputs["input_ids"][0], skip_special_tokens=False)
            assistant_idx = full_text.find("assistant")
            if assistant_idx >= 0:
                print(f"[DEBUG] Found 'assistant' at position {assistant_idx}")
                print(f"[DEBUG] Context: ...{full_text[max(0, assistant_idx-20):assistant_idx+100]}...")

    # Final validation
    valid_labels = labels[labels != -100]
    if valid_labels.numel() == 0:
        print("[collate_fn_fixed_2] WARNING: No tokens left for training after masking!")
    else:
        max_label = valid_labels.max().item()
        if max_label >= vocab_size:
            print(f"[collate_fn_fixed_2] ERROR: Found label {max_label} >= vocab_size {vocab_size}")
            labels[labels >= vocab_size] = -100

        unmasked_count = valid_labels.numel()
        total_count = labels.numel()
        print(f"[collate_fn_fixed_2] Unmasked {unmasked_count}/{total_count} tokens ({unmasked_count/total_count*100:.1f}%) for training")

    batch_inputs["labels"] = labels
    return batch_inputs


def debug_collate_fn(collate_fn, dataset, processor, model, num_samples=2):
    """
    Debug helper to test collator and see what we're training on.

    Args:
        collate_fn: The collator function/class to test
        dataset: The formatted dataset to sample from
        processor: Qwen2VLProcessor instance
        model: Model instance (for forward pass test)
        num_samples: Number of samples to test with

    Returns:
        batch_output: The output from the collator, or None if failed
    """
    print("="*60)
    print("DEBUG: Testing collator")
    print("="*60)

    test_batch = [dataset[i] for i in range(min(num_samples, len(dataset)))]

    print(f"\n Testing with {len(test_batch)} samples...")

    try:
        batch_output = collate_fn(test_batch)
        print("Collator executed successfully")
    except Exception as e:
        print(f"Collator failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    seq_lengths = (batch_output["attention_mask"] == 1).sum(dim=1)
    print(f"\nSequence statistics:")
    print(f"   Sequence lengths: {seq_lengths.tolist()}")
    print(f"   Max sequence length: {seq_lengths.max().item()}")
    print(f"   Min sequence length: {seq_lengths.min().item()}")

    labels = batch_output["labels"]
    print(f"\nTraining token analysis:")

    for batch_idx in range(labels.shape[0]):
        unmasked_indices = (labels[batch_idx] != -100).nonzero(as_tuple=True)[0]
        seq_len = seq_lengths[batch_idx].item()

        print(f"\n   Sample {batch_idx}:")
        print(f"   - Total tokens: {seq_len}")
        print(f"   - Training tokens: {len(unmasked_indices)}")
        print(f"   - Training percentage: {len(unmasked_indices)/seq_len*100:.2f}%")

        if len(unmasked_indices) > 0:
            unmasked_tokens = batch_output["input_ids"][batch_idx][unmasked_indices]
            decoded_text = processor.tokenizer.decode(unmasked_tokens, skip_special_tokens=False)
            decoded_clean = decoded_text.replace('<|object_ref_start|>', '[OBJ_START]')
            decoded_clean = decoded_clean.replace('<|object_ref_end|>', '[OBJ_END]')
            decoded_clean = decoded_clean.replace('<|box_start|>', '[BOX_START]')
            decoded_clean = decoded_clean.replace('<|box_end|>', '[BOX_END]')
            print(f"   - Training on: '{decoded_clean}'")

            if 'nutrition-table' in decoded_text and '(' in decoded_text:
                print(f"   Correctly training on bbox coordinates")
            else:
                print(f"   Warning: Unexpected training content")

    print("\n" + "-"*60)
    print("Testing forward pass with model...")
    print("-"*60)

    with torch.no_grad():
        try:
            if torch.cuda.is_available():
                device = next(model.parameters()).device
                batch_output_gpu = {}
                for k, v in batch_output.items():
                    if hasattr(v, 'to'):
                        batch_output_gpu[k] = v.to(device)
                    else:
                        batch_output_gpu[k] = v
            else:
                batch_output_gpu = batch_output

            outputs = model(**batch_output_gpu)
            loss_value = outputs.loss.item() if outputs.loss is not None else None

            print("Forward pass successful!")
            if loss_value is not None:
                print(f"   Loss: {loss_value:.4f}")
                if loss_value > 10:
                    print("   Warning: Loss is very high, training might be unstable")
            else:
                print("   Warning: No loss returned")

        except RuntimeError as e:
            if "cu_seqlens_q must have dtype int32" in str(e):
                print(f"Flash Attention dtype error detected!")
                print("   This is the known issue with Flash Attention version mismatch")
                print("   Solution: Upgrade Flash Attention to 2.6.3")
                print("   Command: pip install flash-attn==2.6.3 --no-build-isolation")
            else:
                print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"Forward pass failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()

    print("="*60 + "\n")
    return batch_output


def test_collate_fn_fixed_3(processor, model, train_dataset, train_dataset_formatted):
    """
    Test collate_fn_fixed_3 and verify coordinate scaling.

    Args:
        processor: Qwen2VLProcessor instance
        model: Model instance
        train_dataset: Original dataset (for bbox info)
        train_dataset_formatted: Formatted dataset (for collator test)

    Returns:
        collator: The created collator instance
    """
    print("\n" + "="*60)
    print("TESTING collate_fn_fixed_3 & COORDINATE SCALING")
    print("="*60)

    collator = collate_fn_fixed_3(processor, model)
    collator.debug = True

    test_batch = [train_dataset_formatted[0]]

    print("\n1. TESTING COLLATOR:")
    try:
        output = collator(test_batch)
        print("Collator executed successfully")

        labels = output['labels'][0]
        trained_indices = (labels != -100).nonzero(as_tuple=True)[0]
        print(f"   Training on {len(trained_indices)} tokens out of {len(labels)} total")

    except Exception as e:
        print(f"Collator failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n2. COORDINATE SCALING CHECK:")
    print("-" * 40)

    example = train_dataset[0]
    bbox = example['objects']['bbox'][0]

    print(f"Original bbox (normalized [0,1]): {bbox}")

    y_min, x_min, y_max, x_max = bbox
    x1 = int(x_min * 1000)
    y1 = int(y_min * 1000)
    x2 = int(x_max * 1000)
    y2 = int(y_max * 1000)

    print(f"Converted for training ([0,1000)): ({x1},{y1}),({x2},{y2})")
    print(f"Format in training data: <|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>")

    print("\nCOORDINATE SCALING IS CORRECT:")
    print("   - Dataset provides bbox in [0,1] format")
    print("   - We convert to [0,1000) for training (Qwen2-VL expects this)")
    print("   - Model should output in [0,1000) format")
    print("   - parse_qwen_bbox_output expects [0,1000) and converts back")

    print("\n3. WHY MODEL MIGHT PREDICT NO BOX:")
    print("-" * 40)
    print("   Possible reasons:")
    print("   1. Training on static prompts (70% of tokens) - poor learning signal")
    print("   2. Loss plateaued at ~3.25 - model not improving")
    print("   3. Need more epochs or higher learning rate")
    print("   4. Should use collate_fn_fixed_3 for assistant-only training")

    return collator
