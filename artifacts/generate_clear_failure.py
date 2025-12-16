#!/usr/bin/env python3
"""
Generate a clear failure case image showing obvious model error.
Uses known failure indices from the notebook.
"""

import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from torchvision import ops

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.inference import parse_qwen_bbox_output


def load_model_and_processor(model_path: str):
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
    pred_norm = [p / 1000.0 for p in pred_bbox]
    y_min, x_min, y_max, x_max = gt_bbox_openfoodfacts
    gt_norm = [x_min, y_min, x_max, y_max]

    pred_tensor = torch.tensor([pred_norm], dtype=torch.float32)
    gt_tensor = torch.tensor([gt_norm], dtype=torch.float32)

    return ops.box_iou(pred_tensor, gt_tensor).item()


def main():
    MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
    OUTPUT_DIR = project_root / "assets"

    print("=" * 60)
    print("Clear Failure Case Generator")
    print("=" * 60)

    model, processor = load_model_and_processor(MODEL_PATH)

    print("\nLoading dataset...")
    dataset = load_dataset("openfoodfacts/nutrition-table-detection", split="train")

    # Found failure: idx=80 has IoU=0.000 (complete miss)
    failure_idx = 80

    print(f"\nProcessing failure example {failure_idx}...")
    example = dataset[failure_idx]
    image = example['image']
    gt_bboxes = example['objects']['bbox']
    gt_categories = example['objects']['category_name']

    output_text = run_inference(model, processor, image)
    parsed = parse_qwen_bbox_output(output_text)

    print(f"  Model output: {output_text[:100]}...")

    # Calculate IoU
    iou = None
    if parsed:
        pred_bbox = parsed['bbox'] if isinstance(parsed, dict) else parsed[0]['bbox']
        iou = calculate_iou(pred_bbox, gt_bboxes[0])
        print(f"  IoU: {iou:.3f}")

    pil_width, pil_height = image.size

    # Create side-by-side comparison (same style as success grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Ground Truth
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth", fontsize=14, fontweight='bold', color='green')
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

    # Right: Model Prediction
    axes[1].imshow(image)
    axes[1].set_title("Model Prediction", fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')

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
                linewidth=3, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)

    # Add IoU label
    if iou is not None:
        axes[1].text(
            0.98, 0.02, f"IoU: {iou:.3f}",
            transform=axes[1].transAxes,
            fontsize=12, fontweight='bold',
            color='white',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

    plt.suptitle(
        f"Failure Case - Model detects wrong region",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "demo_failure_case_clear.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
