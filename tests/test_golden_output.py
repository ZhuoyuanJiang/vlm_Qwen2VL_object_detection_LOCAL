"""
Golden Output Test - Direct comparison with expected format.

TODO: Complete this test by filling in the expected_format with actual output.

Steps to complete:
1. Load one specific datapoint (e.g., train_dataset[0])
2. Run convert_to_conversation_format on it
3. Print the output and copy it here as expected_format
4. Run this test to verify format stays consistent

This is the MOST INTUITIVE test - direct comparison: result == expected_format
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from datasets import load_dataset
from src.data.dataset import convert_to_conversation_format
from src.data.collators import restore_images_in_conversations


def test_convert_to_conversation_format_golden_output():
    """
    Test convert_to_conversation_format with golden output from train_dataset[0].

    This test directly compares result with expected format for ONE specific datapoint.
    Very simple and intuitive!
    """
    # Load one specific datapoint
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train[:1]", streaming=False)
    sample = ds[0]

    # Expected format from actual output
    expected_messages = [
        {'role': 'system',
         'content': [{'type': 'text',
           'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format.\nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
        {'role': 'user',
         'content': [{'type': 'image', 'image': 'IMAGE_PLACEHOLDER'},
          {'type': 'text',
           'text': 'Detect the bounding box of the nutrition table.'}]},
        {'role': 'assistant',
         'content': [{'type': 'text',
           'text': '<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(14,57),(991,603)<|box_end|>'}]}
    ]

    expected_image_size = (2592, 1944)

    # Convert sample
    result = convert_to_conversation_format(sample)

    # Test messages structure
    assert result['messages'] == expected_messages, \
        f"Messages don't match!\nExpected: {expected_messages}\nGot: {result['messages']}"

    # Test image (can't directly compare PIL images with ==)
    assert isinstance(result['image'], Image.Image), "Image must be PIL.Image object"
    assert result['image'].size == expected_image_size, \
        f"Image size mismatch! Expected: {expected_image_size}, Got: {result['image'].size}"

    print("✅ PASS: convert_to_conversation_format golden output matches!")


def test_restore_images_golden_output():
    """
    Test restore_images_in_conversations output matches expected format exactly.
    """
    # Load 2 samples
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train[:2]", streaming=False)
    formatted_samples = [convert_to_conversation_format(ds[i]) for i in range(2)]
    messages_list = [s['messages'] for s in formatted_samples]
    images_list = [s['image'] for s in formatted_samples]

    # Run the function
    result = restore_images_in_conversations(messages_list, images_list)

    # ============================================================
    # EXPECTED OUTPUT - This is EXACTLY what result should look like:
    # ============================================================
    #
    # [[{'role': 'system',
    #    'content': [{'type': 'text',
    #      'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format.\nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
    #   {'role': 'user',
    #    'content': [{'type': 'image',
    #      'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2592x1944>},
    #     {'type': 'text',
    #      'text': 'Detect the bounding box of the nutrition table.'}]},
    #   {'role': 'assistant',
    #    'content': [{'type': 'text',
    #      'text': '<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(14,57),(991,603)<|box_end|>'}]}],
    #  [{'role': 'system',
    #    'content': [{'type': 'text',
    #      'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format.\nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
    #   {'role': 'user',
    #    'content': [{'type': 'image',
    #      'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=306x408>},
    #     {'type': 'text',
    #      'text': 'Detect the bounding box of the nutrition table.'}]},
    #   {'role': 'assistant',
    #    'content': [{'type': 'text',
    #      'text': '<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(147,151),(516,588)<|box_end|>'}]}]]
    #
    # NOTE: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2592x1944> is Python's
    # display representation of a PIL Image object. You can't write it as code directly.
    # So in expected below, we use images_list[0] which IS that PIL Image object.
    # When you print images_list[0], it shows: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2592x1944>

    expected = [
        # Sample 0
        [{'role': 'system',
          'content': [{'type': 'text',
            'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format.\nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
         {'role': 'user',
          'content': [{'type': 'image',
            'image': images_list[0]},  # = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2592x1944>
           {'type': 'text',
            'text': 'Detect the bounding box of the nutrition table.'}]},
         {'role': 'assistant',
          'content': [{'type': 'text',
            'text': '<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(14,57),(991,603)<|box_end|>'}]}],
        # Sample 1
        [{'role': 'system',
          'content': [{'type': 'text',
            'text': 'You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format.\nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.'}]},
         {'role': 'user',
          'content': [{'type': 'image',
            'image': images_list[1]},  # = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=306x408>
           {'type': 'text',
            'text': 'Detect the bounding box of the nutrition table.'}]},
         {'role': 'assistant',
          'content': [{'type': 'text',
            'text': '<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(147,151),(516,588)<|box_end|>'}]}]
    ]

    # ============================================================
    # VERIFY: result == expected
    # ============================================================
    # Direct comparison works for everything because we use the same PIL Image objects.
    # If restore_images_in_conversations returns the correct images, they'll be identical.

    assert result == expected, f"Output doesn't match expected!\n\nGot:\n{result}\n\nExpected:\n{expected}"

    print("✅ PASS: restore_images_in_conversations output == expected")


if __name__ == "__main__":
    print("="*70)
    print("GOLDEN OUTPUT TESTS")
    print("="*70)

    test_convert_to_conversation_format_golden_output()
    test_restore_images_golden_output()

    print("="*70)
    print("All golden output tests passed!")
    print("="*70)
