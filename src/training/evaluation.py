"""
Evaluation functions for Qwen2-VL object detection models.

This module provides evaluation utilities for measuring model performance
using IoU (Intersection over Union) metrics.

Extracted from: fine_tuning_vlm_for_object_detection_trl.py (lines 2384-2503)

Usage:
    from src.training.evaluation import evaluate_model

    metrics, ious = evaluate_model(model, processor, eval_dataset, num_samples=50)
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
"""

import torch
import numpy as np
from tqdm import tqdm
from torchvision import ops
from qwen_vl_utils import process_vision_info

from ..models.inference import parse_qwen_bbox_output


def evaluate_model(model, processor, dataset, num_samples=50):
    """
    Evaluate the model on a dataset using IoU metric.

    This function runs inference on a subset of the dataset and computes
    IoU (Intersection over Union) between predicted and ground truth bboxes.

    Args:
        model: The model to evaluate (can be base or fine-tuned)
        processor: The processor for the model (Qwen2VLProcessor)
        dataset: The evaluation dataset (must have 'image' and 'objects' fields)
        num_samples: Number of samples to evaluate (default: 50)

    Returns:
        tuple: (metrics_dict, ious_list)
            - metrics_dict: Dictionary with evaluation metrics:
                - mean_iou: Mean IoU across all samples
                - median_iou: Median IoU
                - max_iou: Maximum IoU
                - min_iou: Minimum IoU
                - detection_rate: Percentage of samples with valid detections
                - samples_evaluated: Number of samples evaluated
                - iou_threshold_0.5: Percentage of samples with IoU > 0.5
                - iou_threshold_0.7: Percentage of samples with IoU > 0.7
            - ious_list: List of IoU scores for each sample

    Example:
        >>> from src.training.evaluation import evaluate_model
        >>> metrics, ious = evaluate_model(model_finetuned, processor, eval_dataset, num_samples=50)
        >>> print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        >>> print(f"Detection Rate: {metrics['detection_rate']:.2%}")

    Note:
        - Ground truth bbox format: [y_min, x_min, y_max, x_max] (OpenFoodFacts format)
        - Predicted bbox format: [x_min, y_min, x_max, y_max] in [0, 1000) range
        - IoU calculation converts both to [x_min, y_min, x_max, y_max] normalized format
    """
    ious = []
    successful_detections = 0
    total_samples = min(num_samples, len(dataset))

    for idx in tqdm(range(total_samples), desc="Evaluating"):
        example = dataset[idx]
        image = example['image']
        ground_truth_bbox = example['objects']['bbox'][0]  # Take first bbox

        # Create messages for inference
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Detect the bounding box of the nutrition table."},
                ],
            }
        ]

        try:
            # Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision information
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Parse the model output
            parsed_bbox = parse_qwen_bbox_output(output_text)

            if parsed_bbox:
                successful_detections += 1

                # Convert predicted bbox from Qwen format (1000x1000) to normalized [0,1]
                if isinstance(parsed_bbox, list):
                    pred_bbox = parsed_bbox[0]['bbox']
                else:
                    pred_bbox = parsed_bbox['bbox']

                # Normalize predicted bbox from 1000x1000 to 0-1
                pred_bbox_norm = [
                    pred_bbox[0] / 1000.0,
                    pred_bbox[1] / 1000.0,
                    pred_bbox[2] / 1000.0,
                    pred_bbox[3] / 1000.0
                ]

                # Convert to tensor for IoU calculation
                # CRITICAL: Convert ground_truth from [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max]
                # because torch.ops.box_iou expects [x_min, y_min, x_max, y_max] format
                y_min_gt, x_min_gt, y_max_gt, x_max_gt = ground_truth_bbox
                gt_tensor = torch.tensor([[x_min_gt, y_min_gt, x_max_gt, y_max_gt]], dtype=torch.float32)

                # pred_bbox_norm is already in [x_min, y_min, x_max, y_max] format from Qwen output
                pred_tensor = torch.tensor([[pred_bbox_norm[0], pred_bbox_norm[1],
                                           pred_bbox_norm[2], pred_bbox_norm[3]]], dtype=torch.float32)

                # Calculate IoU
                iou = ops.box_iou(pred_tensor, gt_tensor).item()
                ious.append(iou)
            else:
                ious.append(0.0)  # No detection counts as 0 IoU

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            ious.append(0.0)

    # Calculate metrics
    metrics = {
        "mean_iou": np.mean(ious) if ious else 0.0,
        "median_iou": np.median(ious) if ious else 0.0,
        "max_iou": np.max(ious) if ious else 0.0,
        "min_iou": np.min(ious) if ious else 0.0,
        "detection_rate": successful_detections / total_samples,
        "samples_evaluated": total_samples,
        "iou_threshold_0.5": sum(1 for iou in ious if iou > 0.5) / total_samples,
        "iou_threshold_0.7": sum(1 for iou in ious if iou > 0.7) / total_samples,
    }

    return metrics, ious


def print_evaluation_results(metrics, model_name="Model"):
    """
    Print evaluation results in a formatted way.

    Args:
        metrics: Dictionary of metrics from evaluate_model()
        model_name: Name to display for the model (default: "Model")

    Example:
        >>> metrics, ious = evaluate_model(model, processor, dataset)
        >>> print_evaluation_results(metrics, "Fine-tuned Model")
    """
    print(f"\n{model_name} Evaluation Results:")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Median IoU: {metrics['median_iou']:.4f}")
    print(f"  Detection Rate: {metrics['detection_rate']:.2%}")
    print(f"  IoU > 0.5: {metrics['iou_threshold_0.5']:.2%}")
    print(f"  IoU > 0.7: {metrics['iou_threshold_0.7']:.2%}")


def compare_models(metrics_base, metrics_finetuned):
    """
    Compare evaluation metrics between base and fine-tuned models.

    Args:
        metrics_base: Metrics dictionary for the base model
        metrics_finetuned: Metrics dictionary for the fine-tuned model

    Example:
        >>> metrics_base, _ = evaluate_model(base_model, processor, dataset)
        >>> metrics_ft, _ = evaluate_model(finetuned_model, processor, dataset)
        >>> compare_models(metrics_base, metrics_ft)
    """
    print("\n" + "="*50)
    print("COMPARISON: Fine-tuned vs Base Model")
    print("="*50)

    iou_improvement = metrics_finetuned['mean_iou'] - metrics_base['mean_iou']
    iou_improvement_pct = (iou_improvement / max(metrics_base['mean_iou'], 0.001)) * 100

    print(f"Mean IoU Improvement: {iou_improvement:.4f} ({iou_improvement_pct:.1f}% improvement)")
    print(f"Detection Rate Improvement: {(metrics_finetuned['detection_rate'] - metrics_base['detection_rate']):.2%}")
    print(f"IoU > 0.5 Improvement: {(metrics_finetuned['iou_threshold_0.5'] - metrics_base['iou_threshold_0.5']):.2%}")
