#!/usr/bin/env python3
"""
Generate a single failure case image for README.

Finds an example where the model performs poorly (low IoU)
and creates a comparison image.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from torchvision import ops

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.inference import parse_qwen_bbox_output


def load_model_and_processor(model_path: str):
    """Load the fine-tuned merged model."""
    print(f"Loading model from: {model_path}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = Qwen2VLProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )

    model.eval()
    return model, processor


def run_inference(model, processor, image):
    """Run inference on a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Detect the bounding box of the nutrition table."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


def calculate_iou(pred_bbox, gt_bbox_openfoodfacts):
    """Calculate IoU between prediction and ground truth."""
    pred_norm = [
        pred_bbox[0] / 1000.0,
        pred_bbox[1] / 1000.0,
        pred_bbox[2] / 1000.0,
        pred_bbox[3] / 1000.0,
    ]

    y_min, x_min, y_max, x_max = gt_bbox_openfoodfacts
    gt_norm = [x_min, y_min, x_max, y_max]

    pred_tensor = torch.tensor([pred_norm], dtype=torch.float32)
    gt_tensor = torch.tensor([gt_norm], dtype=torch.float32)

    iou = ops.box_iou(pred_tensor, gt_tensor).item()
    return iou


def main():
    MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
    OUTPUT_DIR = project_root / "assets"

    print("=" * 60)
    print("Failure Case Generator")
    print("=" * 60)

    model, processor = load_model_and_processor(MODEL_PATH)

    print("\nLoading dataset...")
    dataset = load_dataset("openfoodfacts/nutrition-table-detection", split="train")

    # Search for failure cases (low IoU)
    print("\nSearching for failure cases (IoU < 0.5)...")
    failure_cases = []

    # Sample more examples to find failures
    sample_indices = list(range(0, 200, 5))  # Check every 5th example

    for idx in sample_indices:
        if len(failure_cases) >= 5:
            break

        example = dataset[idx]
        image = example['image']
        gt_bboxes = example['objects']['bbox']

        # Only check single-bbox examples for simplicity
        if len(gt_bboxes) != 1:
            continue

        output_text = run_inference(model, processor, image)
        parsed = parse_qwen_bbox_output(output_text)

        if parsed:
            pred_bbox = parsed['bbox'] if isinstance(parsed, dict) else parsed[0]['bbox']
            iou = calculate_iou(pred_bbox, gt_bboxes[0])

            if iou < 0.5:
                failure_cases.append({
                    'idx': idx,
                    'iou': iou,
                    'parsed': parsed,
                    'output_text': output_text,
                })
                print(f"  Found failure: idx={idx}, IoU={iou:.3f}")

    if not failure_cases:
        print("No failure cases found with IoU < 0.5. Searching for lowest IoU...")
        # Find the worst case if no clear failures
        all_cases = []
        for idx in sample_indices[:30]:
            example = dataset[idx]
            if len(example['objects']['bbox']) != 1:
                continue

            output_text = run_inference(model, processor, example['image'])
            parsed = parse_qwen_bbox_output(output_text)

            if parsed:
                pred_bbox = parsed['bbox'] if isinstance(parsed, dict) else parsed[0]['bbox']
                iou = calculate_iou(pred_bbox, example['objects']['bbox'][0])
                all_cases.append({'idx': idx, 'iou': iou, 'parsed': parsed})

        all_cases.sort(key=lambda x: x['iou'])
        failure_cases = all_cases[:1]

    if failure_cases:
        # Use the worst case
        worst = failure_cases[0]
        idx = worst['idx']
        iou = worst['iou']

        print(f"\nGenerating failure case image for idx={idx}, IoU={iou:.3f}")

        example = dataset[idx]
        image = example['image']
        gt_bboxes = example['objects']['bbox']
        gt_categories = example['objects']['category_name']

        output_text = run_inference(model, processor, image)
        parsed = parse_qwen_bbox_output(output_text)

        # Create comparison image
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        pil_width, pil_height = image.size

        # Left: Ground Truth
        axes[0].imshow(image)
        axes[0].set_title("Ground Truth", fontsize=14)
        axes[0].axis('off')

        for bbox, category in zip(gt_bboxes, gt_categories):
            y_min, x_min, y_max, x_max = bbox
            gt_x1 = int(x_min * pil_width)
            gt_y1 = int(y_min * pil_height)
            gt_x2 = int(x_max * pil_width)
            gt_y2 = int(y_max * pil_height)

            rect = Rectangle(
                (gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1,
                linewidth=3, edgecolor='green', facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                gt_x1, gt_y1 - 10, f"GT: {category}",
                color='green', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        # Right: Model Prediction
        axes[1].imshow(image)
        axes[1].set_title(f"Model Prediction - IoU: {iou:.3f}", fontsize=14)
        axes[1].axis('off')

        if parsed:
            pred_list = parsed if isinstance(parsed, list) else [parsed]
            for pred in pred_list:
                bbox = pred['bbox']
                obj_name = pred.get('object', 'nutrition table')

                pred_x1 = int(bbox[0] * pil_width / 1000)
                pred_y1 = int(bbox[1] * pil_height / 1000)
                pred_x2 = int(bbox[2] * pil_width / 1000)
                pred_y2 = int(bbox[3] * pil_height / 1000)

                rect = Rectangle(
                    (pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1,
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                axes[1].add_patch(rect)
                axes[1].text(
                    pred_x1, pred_y1 - 10, f"Pred: {obj_name}",
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

        plt.suptitle(
            f"Failure Case - Example {idx}\n"
            f"Model struggles with this image (IoU: {iou:.3f})",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        output_path = OUTPUT_DIR / "demo_failure_case.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
