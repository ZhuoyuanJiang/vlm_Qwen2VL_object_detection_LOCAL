#!/usr/bin/env python3
"""
Generate Demo Images for README

Creates comparison images showing Ground Truth vs Model Prediction
for the fine-tuned Qwen2-VL nutrition table detection model.

Usage:
    python scripts/generate_demo_images.py

Output:
    Creates multiple images in assets/ folder for README selection.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
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
    print(f"Model loaded: {model.get_memory_footprint() / 1024**3:.2f} GB")
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


def calculate_iou(pred_bbox, gt_bbox_openfoodfacts, image_size):
    """
    Calculate IoU between prediction and ground truth.

    Args:
        pred_bbox: [x1, y1, x2, y2] in Qwen's [0, 1000] space
        gt_bbox_openfoodfacts: [y_min, x_min, y_max, x_max] normalized [0, 1]
        image_size: (width, height) of the image

    Returns:
        IoU score (float)
    """
    # Normalize pred_bbox from 1000x1000 to 0-1
    pred_norm = [
        pred_bbox[0] / 1000.0,
        pred_bbox[1] / 1000.0,
        pred_bbox[2] / 1000.0,
        pred_bbox[3] / 1000.0,
    ]

    # Convert GT from OpenFoodFacts format [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max]
    y_min, x_min, y_max, x_max = gt_bbox_openfoodfacts
    gt_norm = [x_min, y_min, x_max, y_max]

    pred_tensor = torch.tensor([pred_norm], dtype=torch.float32)
    gt_tensor = torch.tensor([gt_norm], dtype=torch.float32)

    iou = ops.box_iou(pred_tensor, gt_tensor).item()
    return iou


def create_comparison_image(
    image,
    gt_bboxes,
    gt_categories,
    pred_bboxes,
    iou_score=None,
    title="",
    output_path=None,
):
    """
    Create side-by-side comparison image: Ground Truth vs Model Prediction.

    Style matches notebooks/02_model_understanding.ipynb
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pil_width, pil_height = image.size

    # Left: Ground Truth
    axes[0].imshow(image)
    axes[0].set_title(f"Ground Truth ({len(gt_bboxes)} box{'es' if len(gt_bboxes) > 1 else ''})", fontsize=14)
    axes[0].axis('off')

    for idx, (bbox, category) in enumerate(zip(gt_bboxes, gt_categories)):
        # OpenFoodFacts format: [y_min, x_min, y_max, x_max] normalized [0,1]
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
    axes[1].axis('off')

    if pred_bboxes:
        pred_list = pred_bboxes if isinstance(pred_bboxes, list) else [pred_bboxes]
        title_text = f"Model Prediction ({len(pred_list)} box{'es' if len(pred_list) > 1 else ''})"
        if iou_score is not None:
            title_text += f" - IoU: {iou_score:.3f}"
        axes[1].set_title(title_text, fontsize=14)

        for idx, pred in enumerate(pred_list):
            bbox = pred['bbox']
            obj_name = pred.get('object', 'nutrition table')

            # Qwen format: [x1, y1, x2, y2] in [0, 1000] space
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
    else:
        axes[1].set_title("Model Prediction - No detection", fontsize=14)

    if title:
        plt.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {output_path}")

    plt.close()
    return fig


def find_examples_by_bbox_count(dataset, target_count, num_examples=3):
    """Find examples with specific number of bounding boxes."""
    found = []
    for idx, example in enumerate(dataset):
        num_bboxes = len(example['objects']['bbox'])
        if num_bboxes == target_count:
            found.append(idx)
            if len(found) >= num_examples:
                break
    return found


def main():
    # Configuration
    MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
    OUTPUT_DIR = project_root / "assets"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Demo Image Generator")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Load model
    model, processor = load_model_and_processor(MODEL_PATH)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("openfoodfacts/nutrition-table-detection", split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # Find examples with different bbox counts
    print("\nFinding example indices...")
    single_bbox_indices = find_examples_by_bbox_count(dataset, 1, num_examples=5)
    two_bbox_indices = find_examples_by_bbox_count(dataset, 2, num_examples=3)
    three_bbox_indices = find_examples_by_bbox_count(dataset, 3, num_examples=2)

    print(f"  Single bbox examples: {single_bbox_indices}")
    print(f"  Two bbox examples: {two_bbox_indices}")
    print(f"  Three bbox examples: {three_bbox_indices}")

    # Generate demo images
    print("\n" + "=" * 60)
    print("Generating Demo Images")
    print("=" * 60)

    # Track best IoU examples for highlighting
    all_results = []

    # 1. Single bbox examples
    print("\n--- Single Bbox Examples ---")
    for i, idx in enumerate(single_bbox_indices[:3]):
        example = dataset[idx]
        image = example['image']
        gt_bboxes = example['objects']['bbox']
        gt_categories = example['objects']['category_name']

        print(f"\nProcessing example {idx}...")
        output_text = run_inference(model, processor, image)
        parsed = parse_qwen_bbox_output(output_text)

        iou = None
        if parsed:
            pred_bbox = parsed['bbox'] if isinstance(parsed, dict) else parsed[0]['bbox']
            iou = calculate_iou(pred_bbox, gt_bboxes[0], image.size)
            print(f"  IoU: {iou:.3f}")
        else:
            print(f"  No detection")

        output_path = OUTPUT_DIR / f"demo_single_bbox_{i+1}.png"
        create_comparison_image(
            image, gt_bboxes, gt_categories, parsed,
            iou_score=iou,
            title=f"Nutrition Table Detection - Example {idx}",
            output_path=output_path
        )

        all_results.append({
            'idx': idx,
            'type': 'single',
            'iou': iou,
            'path': output_path,
        })

    # 2. Two bbox examples
    print("\n--- Two Bbox Examples ---")
    for i, idx in enumerate(two_bbox_indices[:2]):
        example = dataset[idx]
        image = example['image']
        gt_bboxes = example['objects']['bbox']
        gt_categories = example['objects']['category_name']

        print(f"\nProcessing example {idx}...")
        output_text = run_inference(model, processor, image)
        parsed = parse_qwen_bbox_output(output_text)

        print(f"  Model output: {output_text[:100]}...")

        output_path = OUTPUT_DIR / f"demo_multi_bbox_2boxes_{i+1}.png"
        create_comparison_image(
            image, gt_bboxes, gt_categories, parsed,
            title=f"Multi-Object Detection - Example {idx} (2 tables)",
            output_path=output_path
        )

        all_results.append({
            'idx': idx,
            'type': '2bbox',
            'path': output_path,
        })

    # 3. Three bbox examples
    print("\n--- Three Bbox Examples ---")
    for i, idx in enumerate(three_bbox_indices[:1]):
        example = dataset[idx]
        image = example['image']
        gt_bboxes = example['objects']['bbox']
        gt_categories = example['objects']['category_name']

        print(f"\nProcessing example {idx}...")
        output_text = run_inference(model, processor, image)
        parsed = parse_qwen_bbox_output(output_text)

        print(f"  Model output: {output_text[:100]}...")

        output_path = OUTPUT_DIR / f"demo_multi_bbox_3boxes.png"
        create_comparison_image(
            image, gt_bboxes, gt_categories, parsed,
            title=f"Multi-Object Detection - Example {idx} (3 tables)",
            output_path=output_path
        )

        all_results.append({
            'idx': idx,
            'type': '3bbox',
            'path': output_path,
        })

    # 4. Create a grid comparison image
    print("\n--- Creating Comparison Grid ---")

    # Get best 4 single-bbox examples by IoU
    single_results = [r for r in all_results if r['type'] == 'single' and r['iou'] is not None]
    single_results.sort(key=lambda x: x['iou'], reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, result in enumerate(single_results[:4]):
        idx = result['idx']
        example = dataset[idx]
        image = example['image']
        gt_bboxes = example['objects']['bbox']

        output_text = run_inference(model, processor, image)
        parsed = parse_qwen_bbox_output(output_text)

        axes[i].imshow(image)
        axes[i].axis('off')

        pil_width, pil_height = image.size

        # Draw GT (green)
        for bbox in gt_bboxes:
            y_min, x_min, y_max, x_max = bbox
            gt_x1 = int(x_min * pil_width)
            gt_y1 = int(y_min * pil_height)
            gt_x2 = int(x_max * pil_width)
            gt_y2 = int(y_max * pil_height)
            rect = Rectangle(
                (gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            axes[i].add_patch(rect)

        # Draw prediction (red)
        if parsed:
            pred_list = parsed if isinstance(parsed, list) else [parsed]
            for pred in pred_list:
                bbox = pred['bbox']
                pred_x1 = int(bbox[0] * pil_width / 1000)
                pred_y1 = int(bbox[1] * pil_height / 1000)
                pred_x2 = int(bbox[2] * pil_width / 1000)
                pred_y2 = int(bbox[3] * pil_height / 1000)
                rect = Rectangle(
                    (pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[i].add_patch(rect)

        iou = result['iou']
        axes[i].set_title(f"Example {idx} - IoU: {iou:.3f}", fontsize=12)

    plt.suptitle(
        "Fine-tuned Qwen2-VL Nutrition Table Detection\n"
        "Green (dashed): Ground Truth | Red (solid): Model Prediction",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    grid_path = OUTPUT_DIR / "demo_comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {grid_path}")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated images in: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

    print("\n" + "-" * 60)
    print("IoU Summary for single-bbox examples:")
    for r in all_results:
        if r['type'] == 'single' and r['iou'] is not None:
            print(f"  Example {r['idx']}: IoU = {r['iou']:.3f}")

    print("\nRecommended for README:")
    if single_results:
        best = single_results[0]
        print(f"  - Best single bbox: demo_single_bbox_*.png (IoU: {best['iou']:.3f})")
    print("  - Multi-object demo: demo_multi_bbox_2boxes_*.png")
    print("  - Grid overview: demo_comparison_grid.png")


if __name__ == "__main__":
    main()
