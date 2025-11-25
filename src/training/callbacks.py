"""
Training callbacks for Qwen2-VL fine-tuning.

This module provides training callbacks for monitoring and evaluation:
- GradNormCallback: Logs gradient L2 norm before optimizer step
- IoUEvalCallback: Runs IoU evaluation at each evaluate event
- ConsoleLogCallback: Prints selected metrics to console

Extracted from: fine_tuning_vlm_for_object_detection_trl.py
"""

import torch
from transformers import TrainerCallback
from qwen_vl_utils import process_vision_info

from ..models.inference import parse_qwen_bbox_output


class GradNormCallback(TrainerCallback):
    """Logs gradient L2 norm over trainable parameters before optimizer step."""

    def __init__(self, log_every_steps=0):
        self.trainer = None
        self.log_every_steps = log_every_steps  # 0 -> use args.logging_steps

    def on_init_end(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Compute gradient statistics BEFORE optimizer.step() and zero_grad()."""
        if self.trainer is None:
            return control
        logging_frequency = self.log_every_steps or getattr(args, "logging_steps", 10)
        if state.global_step % max(logging_frequency, 1) != 0 or state.global_step == 0:
            return control

        total_squared_norm = 0.0
        param_count = 0
        nonzero_grad_count = 0

        # Accumulate L2 norm over all trainable parameters with existing gradients
        for param_name, param_tensor in self.trainer.model.named_parameters():
            if param_tensor.requires_grad and (param_tensor.grad is not None):
                grad = param_tensor.grad.detach()
                gnorm = grad.float().norm(2).item()
                total_squared_norm += gnorm * gnorm
                param_count += 1
                if gnorm > 0:
                    nonzero_grad_count += 1

        pre_clip_norm = (total_squared_norm ** 0.5) if param_count > 0 else 0.0

        max_grad_norm = float(getattr(args, "max_grad_norm", 1.0) or 0.0)
        grad_clipped = int(pre_clip_norm > max_grad_norm) if max_grad_norm > 0 else 0
        clip_ratio = float(min(1.0, max_grad_norm / (pre_clip_norm + 1e-6))) if pre_clip_norm > 0 and max_grad_norm > 0 else 1.0

        gradient_logs = {
            # Keep the original key for continuity in dashboards
            "grad_lora_norm": float(pre_clip_norm),
            # Additional diagnostics
            "grad_norm_pre_clip": float(pre_clip_norm),
            "grad_clipped": int(grad_clipped),
            "grad_clip_ratio": float(clip_ratio),
            "grads_nonzero": int(nonzero_grad_count),
            "grads_total": int(param_count),
        }

        try:
            self.trainer.log(gradient_logs)
        except Exception as _e:
            # Best-effort fallback
            try:
                import wandb as _wandb
                if getattr(_wandb, "run", None) is not None:
                    _wandb.log(gradient_logs, step=int(state.global_step))
                else:
                    print(f"[GradNormCallback] {gradient_logs}")
            except Exception as _e2:
                print(f"[GradNormCallback] Logging failed: {_e}; Fallback failed: {_e2}")
        return control


class IoUEvalCallback(TrainerCallback):
    """Runs quick IoU evaluation at each evaluate event and logs results."""

    def __init__(self, processor, eval_dataset, num_samples=24, prefix="eval"):
        self.trainer = None
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.prefix = prefix

    def on_init_end(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Use the model from the attached trainer; CallbackHandler does not pass model here
        if self.trainer is None or getattr(self.trainer, "model", None) is None:
            return control
        model = self.trainer.model
        logs = None
        try:
            # Lightweight internal IoU eval to avoid dependency ordering issues
            import numpy as _np
            from torchvision import ops as _ops
            num_eval_samples = min(self.num_samples, len(self.eval_dataset))
            iou_scores = []
            successful_predictions = 0
            for sample_idx in range(num_eval_samples):
                dataset_example = self.eval_dataset[sample_idx]
                input_image = dataset_example['image']
                ground_truth_bbox = dataset_example['objects']['bbox'][0]
                # Build messages
                chat_messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": input_image},
                        {"type": "text", "text": "Detect the bounding box of the nutrition table."},
                    ],
                }]
                prompt_text = self.processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
                processed_images, processed_videos = process_vision_info(chat_messages)
                model_inputs = self.processor(text=[prompt_text], images=processed_images, videos=processed_videos, padding=True, return_tensors="pt").to(model.device)
                # Ensure eval mode and no grad for generation
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    generated_output = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
                if was_training:
                    model.train()
                trimmed_outputs = [output[len(input_ids):] for input_ids, output in zip(model_inputs.input_ids, generated_output)]
                decoded_text = self.processor.batch_decode(trimmed_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                parsed_bbox = parse_qwen_bbox_output(decoded_text)
                if parsed_bbox:
                    successful_predictions += 1
                    bbox_coords = parsed_bbox[0]['bbox'] if isinstance(parsed_bbox, list) else parsed_bbox['bbox']
                    predicted_tensor = torch.tensor([[bbox_coords[0]/1000.0, bbox_coords[1]/1000.0, bbox_coords[2]/1000.0, bbox_coords[3]/1000.0]], dtype=torch.float32)
                    y_min, x_min, y_max, x_max = ground_truth_bbox
                    ground_truth_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
                    iou_value = _ops.box_iou(predicted_tensor, ground_truth_tensor).item()
                    iou_scores.append(iou_value)
                else:
                    iou_scores.append(0.0)
            mean_iou = float(_np.mean(iou_scores)) if iou_scores else 0.0
            median_iou = float(_np.median(iou_scores)) if iou_scores else 0.0
            logs = {
                f"{self.prefix}_iou_mean": float(mean_iou),
                f"{self.prefix}_iou_median": float(median_iou),
                f"{self.prefix}_iou_0.5": float(sum(1 for iou_val in iou_scores if iou_val > 0.5) / max(num_eval_samples, 1)),
                f"{self.prefix}_iou_0.7": float(sum(1 for iou_val in iou_scores if iou_val > 0.7) / max(num_eval_samples, 1)),
                f"{self.prefix}_det_rate": float(successful_predictions / max(num_eval_samples, 1)),
            }
            try:
                self.trainer.log(logs)
            except Exception as _e:
                # Fallback to direct W&B logging if available
                try:
                    import wandb as _wandb
                    if getattr(_wandb, "run", None) is not None:
                        _wandb.log(logs, step=int(state.global_step))
                        print(f"[IoUEvalCallback] Logged to W&B directly: {logs}")
                    else:
                        print(f"[IoUEvalCallback] Metrics: {logs}")
                except Exception as _e2:
                    print(f"[IoUEvalCallback] trainer.log failed: {_e}; fallback failed: {_e2}; metrics: {logs}")
        except Exception as e:
            # If evaluation itself fails, report and continue
            print(f"[IoUEvalCallback] eval failed: {e}")
        return control


class ConsoleLogCallback(TrainerCallback):
    """Prints selected metrics to console so readers see them in notebook output."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        keys = [
            "eval_iou_mean", "eval_iou_median", "eval_iou_0.5", "eval_iou_0.7", "eval_det_rate",
            "grad_lora_norm", "grads_nonzero", "grads_total",
        ]
        present = {k: logs[k] for k in keys if k in logs}
        if present:
            try:
                pretty = " ".join(
                    [f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in present.items()]
                )
            except Exception:
                pretty = str(present)
            print(f"[metrics] {pretty}")
        return control
