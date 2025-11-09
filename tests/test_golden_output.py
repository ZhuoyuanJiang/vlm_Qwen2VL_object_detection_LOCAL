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
    TODO: Fill in expected_format with actual output from train_dataset[0]

    This test directly compares result with expected format for ONE specific datapoint.
    Very simple and intuitive!
    """
    # Load one specific datapoint
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
    sample = next(iter(ds))  # First sample

    # TODO: Run convert_to_conversation_format(sample) once,
    #       copy the output here, then uncomment and run test

    expected_format = {
        # TODO: Fill this with actual output
        # Should look like:
        # 'messages': [
        #     {'role': 'system', 'content': [{'type': 'text', 'text': '...'}]},
        #     {'role': 'user', 'content': [
        #         {'type': 'image', 'image': 'IMAGE_PLACEHOLDER'},
        #         {'type': 'text', 'text': 'Detect the bounding box...'}
        #     ]},
        #     {'role': 'assistant', 'content': [{'type': 'text', 'text': '<|object_ref_start|>...'}]}
        # ],
        # 'image': <PIL.Image object - you'll need to handle this separately>
    }

    # Convert sample
    result = convert_to_conversation_format(sample)

    # TODO: Once expected_format is filled, uncomment this:
    # assert result['messages'] == expected_format['messages'], \
    #     f"Messages don't match!\nExpected: {expected_format['messages']}\nGot: {result['messages']}"

    # Note: Can't directly compare PIL images with ==, so check separately:
    # assert isinstance(result['image'], Image.Image)
    # assert result['image'].size == expected_format['image'].size  # if you store size

    print("TODO: Fill in expected_format and uncomment assertions")


def test_restore_images_golden_output():
    """
    TODO: Fill in expected format for restore_images_in_conversations output

    This tests the format AFTER restoring images (what goes into apply_chat_template)
    """
    # Load samples
    ds = load_dataset("openfoodfacts/nutrition-table-detection", split="train", streaming=True)
    samples = [next(iter(ds)) for _ in range(2)]  # Take 2 samples

    # Convert to conversation format
    formatted_samples = [convert_to_conversation_format(s) for s in samples]
    messages_list = [s['messages'] for s in formatted_samples]
    images_list = [s['image'] for s in formatted_samples]

    # TODO: Run restore_images_in_conversations once, copy output here
    expected_format = [
        # TODO: Fill with actual output
        # Should be list of conversations with actual PIL images (not 'IMAGE_PLACEHOLDER')
        # [
        #     [  # Conversation 1
        #         {'role': 'system', 'content': [...]},
        #         {'role': 'user', 'content': [
        #             {'type': 'image', 'image': <PIL.Image>},  # ← Actual PIL image!
        #             {'type': 'text', 'text': '...'}
        #         ]},
        #         {'role': 'assistant', 'content': [...]}
        #     ],
        #     [  # Conversation 2
        #         ...
        #     ]
        # ]
    ]

    # Restore images
    result = restore_images_in_conversations(messages_list, images_list)

    # TODO: Once expected_format is filled, uncomment:
    # Check structure matches
    # assert len(result) == len(expected_format)
    # for i, (actual_conv, expected_conv) in enumerate(zip(result, expected_format)):
    #     assert len(actual_conv) == 3, f"Conversation {i} should have 3 messages"
    #     # Compare roles
    #     assert [msg['role'] for msg in actual_conv] == [msg['role'] for msg in expected_conv]
    #     # Check images are PIL Images (can't directly compare)
    #     for msg in actual_conv:
    #         if msg['role'] == 'user':
    #             image_item = [c for c in msg['content'] if c.get('type') == 'image'][0]
    #             assert isinstance(image_item['image'], Image.Image), "Must be PIL Image!"

    print("TODO: Fill in expected_format and uncomment assertions")


if __name__ == "__main__":
    print("="*70)
    print("GOLDEN OUTPUT TESTS - TODO")
    print("="*70)
    print("\nTo complete these tests:")
    print("1. Run the functions manually and print output")
    print("2. Copy the output into expected_format")
    print("3. Uncomment the assertions")
    print("4. Run tests to verify format stays consistent")
    print("\nExample to get output:")
    print("  ds = load_dataset('openfoodfacts/nutrition-table-detection', split='train', streaming=True)")
    print("  sample = next(iter(ds))")
    print("  result = convert_to_conversation_format(sample)")
    print("  print(result)")
    print("="*70)

    test_convert_to_conversation_format_golden_output()
    test_restore_images_golden_output()
