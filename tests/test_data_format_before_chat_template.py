"""
Test Qwen2-VL Compatibility - verify data format works with actual Qwen2-VL functions.

This test verifies our data format is compatible with:
- processor.apply_chat_template()
- process_vision_info()
- processor() tokenization (full pipeline)

NOTE: For exact value testing of convert_to_conversation_format() and
restore_images_in_conversations(), see tests/test_golden_output.py

Run with:
    pytest tests/test_data_format_before_chat_template.py -v
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

from src.data.dataset import convert_to_conversation_format
from src.data.collators import restore_images_in_conversations


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
    print("QWEN2-VL COMPATIBILITY TESTS")
    print("=" * 70)
    print("\nThese tests verify our data format works with actual Qwen2-VL functions.")
    print("For exact value tests, see: tests/test_golden_output.py\n")

    print("Loading Qwen2-VL processor...")
    from transformers import Qwen2VLProcessor
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    test = TestChatTemplateCompatibility()
    test.test_apply_chat_template(processor)
    test.test_process_vision_info(processor)
    test.test_full_pipeline(processor)

    print("\n" + "=" * 70)
    print("✅ ALL QWEN2-VL COMPATIBILITY TESTS PASSED!")
    print("=" * 70)
