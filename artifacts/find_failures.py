#!/usr/bin/env python3
"""Find actual failure cases by scanning more examples."""

import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from torchvision import ops

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.inference import parse_qwen_bbox_output


def load_model_and_processor(model_path: str):
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def calculate_iou(pred_bbox, gt_bbox_openfoodfacts):
    pred_norm = [p / 1000.0 for p in pred_bbox]
    y_min, x_min, y_max, x_max = gt_bbox_openfoodfacts
    gt_norm = [x_min, y_min, x_max, y_max]
    return ops.box_iou(torch.tensor([pred_norm]), torch.tensor([gt_norm])).item()


def main():
    MODEL_PATH = "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"

    model, processor = load_model_and_processor(MODEL_PATH)
    dataset = load_dataset("openfoodfacts/nutrition-table-detection", split="train")

    print("Scanning for failure cases (IoU < 0.5)...")
    failures = []

    # Check specific indices and random samples
    indices_to_check = [12, 14, 26, 50, 80, 165] + list(range(0, 300, 10))

    for idx in indices_to_check:
        if idx >= len(dataset):
            continue
        example = dataset[idx]
        if len(example['objects']['bbox']) != 1:
            continue

        output_text = run_inference(model, processor, example['image'])
        parsed = parse_qwen_bbox_output(output_text)

        if parsed:
            pred_bbox = parsed['bbox'] if isinstance(parsed, dict) else parsed[0]['bbox']
            iou = calculate_iou(pred_bbox, example['objects']['bbox'][0])

            if iou < 0.5:
                print(f"  FAILURE: idx={idx}, IoU={iou:.3f}")
                failures.append((idx, iou))
            elif iou < 0.7:
                print(f"  Low: idx={idx}, IoU={iou:.3f}")
        else:
            print(f"  NO DETECTION: idx={idx}")
            failures.append((idx, 0.0))

    print(f"\nFound {len(failures)} failure cases:")
    for idx, iou in sorted(failures, key=lambda x: x[1]):
        print(f"  idx={idx}: IoU={iou:.3f}")


if __name__ == "__main__":
    main()
