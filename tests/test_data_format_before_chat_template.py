"""
Test data format BEFORE applying chat template.

This test suite validates that the data format at each stage is correct
for Qwen2-VL's apply_chat_template and process_vision_info functions.

CRITICAL: These tests verify the format that causes training bugs!
If tests pass → format is correct
If tests fail → we found the bug!

Test Flow:
1. Test convert_to_conversation_format() creates correct structure with IMAGE_PLACEHOLDER
2. Test restore_images_in_conversations() replaces placeholder with actual PIL images
3. Test compatibility with apply_chat_template and process_vision_info (MOST IMPORTANT!)

Run with:
    pytest tests/test_data_format_before_chat_template.py -v
    # Or just this file:
    python tests/test_data_format_before_chat_template.py
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image
from datasets import load_dataset

from src.data.dataset import convert_to_conversation_format, _has_image
from src.data.collators import restore_images_in_conversations


class TestConvertToConversationFormat:
    """Test dataset.map() preprocessing creates correct structure."""

    def test_basic_structure(self):
        """Test that convert_to_conversation_format creates correct structure."""
        # Load one sample
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        sample = next(iter(ds))

        # Convert to conversation format
        result = convert_to_conversation_format(sample)

        # Assertions
        assert 'messages' in result, "Result must have 'messages' key"
        assert 'image' in result, "Result must have 'image' key"
        assert len(result['messages']) == 3, "Must have 3 messages (system, user, assistant)"

        print("✅ Basic structure is correct")

    def test_image_placeholder_used(self):
        """Test that IMAGE_PLACEHOLDER is used instead of actual PIL image."""
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        sample = next(iter(ds))

        result = convert_to_conversation_format(sample)

        # Check IMAGE_PLACEHOLDER is used in user message
        user_msg = result['messages'][1]
        assert user_msg['role'] == 'user', "Second message should be user"

        image_content = [c for c in user_msg['content'] if c.get('type') == 'image']
        assert len(image_content) == 1, "User message should have exactly 1 image"
        assert image_content[0]['image'] == 'IMAGE_PLACEHOLDER', \
            "Image should be 'IMAGE_PLACEHOLDER' string, not actual PIL image"

        # Check actual image is stored at top level
        assert isinstance(result['image'], Image.Image), \
            "Actual image should be stored at top level as PIL.Image"

        print("✅ IMAGE_PLACEHOLDER is used correctly")

    def test_message_roles(self):
        """Test that message roles are correct (system, user, assistant)."""
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        sample = next(iter(ds))

        result = convert_to_conversation_format(sample)
        messages = result['messages']

        assert messages[0]['role'] == 'system', "First message should be system"
        assert messages[1]['role'] == 'user', "Second message should be user"
        assert messages[2]['role'] == 'assistant', "Third message should be assistant"

        print("✅ Message roles are correct")

    def test_bbox_format(self):
        """Test that bounding box is converted to Qwen2-VL format."""
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        sample = next(iter(ds))

        result = convert_to_conversation_format(sample)
        assistant_msg = result['messages'][2]
        assistant_text = assistant_msg['content'][0]['text']

        # Check for Qwen2-VL special tokens
        assert '<|object_ref_start|>' in assistant_text, "Should have object_ref_start token"
        assert '<|object_ref_end|>' in assistant_text, "Should have object_ref_end token"
        assert '<|box_start|>' in assistant_text, "Should have box_start token"
        assert '<|box_end|>' in assistant_text, "Should have box_end token"

        # Check coordinate format (x,y),(x,y)
        import re
        bbox_pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
        matches = re.findall(bbox_pattern, assistant_text)
        assert len(matches) > 0, "Should have at least one bounding box in format (x1,y1),(x2,y2)"

        # Check coordinates are in [0, 1000) range
        for match in matches:
            coords = [int(c) for c in match]
            assert all(0 <= c < 1000 for c in coords), \
                f"Coordinates should be in [0, 1000) range, got {coords}"

        print(f"✅ Bounding box format is correct: {assistant_text[:100]}...")


class TestRestoreImagesInConversations:
    """Test collator helper function creates correct format for chat template."""

    def test_restores_actual_images(self):
        """Test that IMAGE_PLACEHOLDER is replaced with actual PIL images."""
        # Setup: Create sample data
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        samples = [next(iter(ds)) for _ in range(3)]

        # Step 1: Convert to conversation format (with IMAGE_PLACEHOLDER)
        formatted_samples = [convert_to_conversation_format(s) for s in samples]

        # Step 2: Extract messages and images
        messages_list = [s['messages'] for s in formatted_samples]
        images_list = [s['image'] for s in formatted_samples]

        # Step 3: Restore images (this is what collator does)
        all_conversations = restore_images_in_conversations(messages_list, images_list)

        # CRITICAL ASSERTION: Check images are restored
        for i, conversation in enumerate(all_conversations):
            user_msg = conversation[1]  # User message should have image
            assert user_msg['role'] == 'user'

            image_content = [c for c in user_msg['content'] if c.get('type') == 'image']
            assert len(image_content) == 1, f"Conversation {i}: User message should have 1 image"

            # CRITICAL: Must be PIL Image, not placeholder!
            actual_image = image_content[0]['image']
            assert isinstance(actual_image, Image.Image), \
                f"Conversation {i}: Image must be PIL.Image, got {type(actual_image)}"
            assert actual_image != 'IMAGE_PLACEHOLDER', \
                f"Conversation {i}: IMAGE_PLACEHOLDER must be replaced!"

        print("✅ Images are restored correctly (PIL.Image, not placeholder)")

    def test_no_none_values(self):
        """Test that no None values exist in content."""
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        samples = [next(iter(ds)) for _ in range(3)]

        formatted_samples = [convert_to_conversation_format(s) for s in samples]
        messages_list = [s['messages'] for s in formatted_samples]
        images_list = [s['image'] for s in formatted_samples]

        all_conversations = restore_images_in_conversations(messages_list, images_list)

        # Check no None values
        for i, conversation in enumerate(all_conversations):
            for msg in conversation:
                assert msg['content'] is not None, f"Conversation {i}: content cannot be None"
                assert len(msg['content']) > 0, f"Conversation {i}: content cannot be empty"

                for content_item in msg['content']:
                    assert content_item is not None, \
                        f"Conversation {i}: content items cannot be None"

                    if content_item.get('type') == 'text':
                        text = content_item.get('text')
                        assert text is not None, f"Conversation {i}: text cannot be None"
                        assert text != 'None', f"Conversation {i}: text cannot be string 'None'"

        print("✅ No None values in content")

    def test_structure_for_chat_template(self):
        """Test that output structure matches what chat template expects."""
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        samples = [next(iter(ds)) for _ in range(3)]

        formatted_samples = [convert_to_conversation_format(s) for s in samples]
        messages_list = [s['messages'] for s in formatted_samples]
        images_list = [s['image'] for s in formatted_samples]

        all_conversations = restore_images_in_conversations(messages_list, images_list)

        # Check structure
        assert isinstance(all_conversations, list), "all_conversations must be a list"
        assert len(all_conversations) == 3, "Should have 3 conversations"

        for i, conversation in enumerate(all_conversations):
            assert isinstance(conversation, list), f"Conversation {i} must be a list"
            assert len(conversation) == 3, \
                f"Conversation {i} must have 3 messages (system, user, assistant)"

            # Check each message structure
            for msg in conversation:
                assert 'role' in msg, f"Conversation {i}: Message must have 'role'"
                assert msg['role'] in ['system', 'user', 'assistant'], \
                    f"Conversation {i}: Invalid role: {msg['role']}"
                assert 'content' in msg, f"Conversation {i}: Message must have 'content'"
                assert isinstance(msg['content'], list), \
                    f"Conversation {i}: Content must be a list"

        print("✅ Structure is correct for chat template")


class TestChatTemplateCompatibility:
    """Test that format works with actual Qwen2-VL apply_chat_template and process_vision_info.

    THIS IS THE MOST IMPORTANT TEST - if this passes, training should work!
    """

    @pytest.fixture(scope="class")
    def processor(self):
        """Load Qwen2-VL processor once for all tests."""
        from transformers import Qwen2VLProcessor
        return Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def test_apply_chat_template(self, processor):
        """Test that apply_chat_template works without errors."""
        from qwen_vl_utils import process_vision_info

        # Get formatted conversations
        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        samples = [next(iter(ds)) for _ in range(2)]

        formatted_samples = [convert_to_conversation_format(s) for s in samples]
        messages_list = [s['messages'] for s in formatted_samples]
        images_list = [s['image'] for s in formatted_samples]

        all_conversations = restore_images_in_conversations(messages_list, images_list)

        try:
            # Step 1: Apply chat template - this should NOT error!
            text = processor.apply_chat_template(
                all_conversations,
                tokenize=False,
                add_generation_prompt=False
            )

            assert isinstance(text, list), "apply_chat_template should return list"
            assert len(text) == len(all_conversations), "Should have same number of texts"

            print("✅ apply_chat_template works!")
            print(f"   Sample output (first 200 chars): {text[0][:200]}...")

        except Exception as e:
            pytest.fail(f"apply_chat_template failed: {e}")

    def test_process_vision_info(self, processor):
        """Test that process_vision_info extracts images correctly."""
        from qwen_vl_utils import process_vision_info

        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        samples = [next(iter(ds)) for _ in range(2)]

        formatted_samples = [convert_to_conversation_format(s) for s in samples]
        messages_list = [s['messages'] for s in formatted_samples]
        images_list = [s['image'] for s in formatted_samples]

        all_conversations = restore_images_in_conversations(messages_list, images_list)

        try:
            # Step 2: Process vision info - this should NOT error!
            images, videos = process_vision_info(all_conversations)

            assert images is not None, "process_vision_info should extract images"
            assert len(images) == len(all_conversations), \
                "Should extract image for each conversation"

            # Check images are PIL Images
            for i, img in enumerate(images):
                assert isinstance(img, Image.Image), \
                    f"Image {i} should be PIL.Image, got {type(img)}"

            print("✅ process_vision_info works!")
            print(f"   Extracted {len(images)} images")

        except Exception as e:
            pytest.fail(f"process_vision_info failed: {e}")

    def test_full_pipeline(self, processor):
        """Test the complete pipeline: chat template → process vision → tokenize.

        THIS IS THE ULTIMATE TEST - if this passes, collator will work in training!
        """
        from qwen_vl_utils import process_vision_info

        ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
        samples = [next(iter(ds)) for _ in range(2)]

        formatted_samples = [convert_to_conversation_format(s) for s in samples]
        messages_list = [s['messages'] for s in formatted_samples]
        images_list = [s['image'] for s in formatted_samples]

        all_conversations = restore_images_in_conversations(messages_list, images_list)

        try:
            # Full pipeline
            text = processor.apply_chat_template(
                all_conversations,
                tokenize=False,
                add_generation_prompt=False
            )

            images, videos = process_vision_info(all_conversations)

            batch_inputs = processor(
                text=text,
                images=images,
                padding=True,
                truncation=False,
                return_tensors="pt"
            )

            # Verify batch_inputs has expected keys
            assert 'input_ids' in batch_inputs, "Should have input_ids"
            assert 'attention_mask' in batch_inputs, "Should have attention_mask"
            assert 'pixel_values' in batch_inputs, "Should have pixel_values"

            print("✅ FULL PIPELINE WORKS! Format is correct for training!")
            print(f"   Batch size: {batch_inputs['input_ids'].shape[0]}")
            print(f"   Sequence length: {batch_inputs['input_ids'].shape[1]}")
            print(f"   Pixel values shape: {batch_inputs['pixel_values'].shape}")

        except Exception as e:
            pytest.fail(f"Full pipeline failed: {e}\n"
                       f"This means the format is WRONG and will cause training bugs!")


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING DATA FORMAT BEFORE CHAT TEMPLATE")
    print("=" * 70)
    print("\nThese tests verify the data format is correct for Qwen2-VL training.")
    print("If all tests pass → format is correct")
    print("If tests fail → we found the bug!\n")

    # Run tests without pytest (for quick debugging)
    print("\n" + "=" * 70)
    print("TEST 1: convert_to_conversation_format()")
    print("=" * 70)
    test1 = TestConvertToConversationFormat()
    test1.test_basic_structure()
    test1.test_image_placeholder_used()
    test1.test_message_roles()
    test1.test_bbox_format()

    print("\n" + "=" * 70)
    print("TEST 2: restore_images_in_conversations()")
    print("=" * 70)
    test2 = TestRestoreImagesInConversations()
    test2.test_restores_actual_images()
    test2.test_no_none_values()
    test2.test_structure_for_chat_template()

    print("\n" + "=" * 70)
    print("TEST 3: Chat Template Compatibility (MOST IMPORTANT!)")
    print("=" * 70)
    from transformers import Qwen2VLProcessor
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    test3 = TestChatTemplateCompatibility()
    test3.test_apply_chat_template(processor)
    test3.test_process_vision_info(processor)
    test3.test_full_pipeline(processor)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nData format is CORRECT for Qwen2-VL training!")
