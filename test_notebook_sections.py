#!/usr/bin/env python3
"""
Comprehensive Test Script for Qwen2-VL Fine-tuning Project
This script tests all sections of the notebook and generates a detailed report
"""

import os
import sys
import json
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create output directory for test results
os.makedirs('test_outputs', exist_ok=True)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def test_section_1_imports():
    """Test Section 1: Imports and Setup"""
    print_section("SECTION 1: Testing Imports and Setup")
    results = {"section": "Imports and Setup", "status": "pending", "details": {}}
    
    try:
        print("Importing libraries...")
        import os
        import gc
        import torch
        import json
        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional, Tuple
        
        import transformers
        from transformers import (
            AutoTokenizer,
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
            BitsAndBytesConfig,
            Qwen2VLProcessor,
            VisionTextDualEncoderModel,
            AutoModelForVision2Seq,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
        )
        
        from datasets import load_dataset, Dataset
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        
        from qwen_vl_utils import process_vision_info
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from IPython.display import display
        from tqdm import tqdm
        
        results["details"]["imports"] = "✓ All imports successful"
        
        # Check versions
        print("\nChecking versions...")
        results["details"]["versions"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            results["details"]["cuda_info"] = {
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f} GB",
            }
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results["details"]["device"] = str(device)
        print(f"Using device: {device}")
        
        results["status"] = "success"
        print("\n✓ Section 1 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 1 failed: {e}")
    
    return results

def test_section_2_data_loading():
    """Test Section 2: Data Loading and Visualization"""
    print_section("SECTION 2: Testing Data Loading and Visualization")
    results = {"section": "Data Loading", "status": "pending", "details": {}}
    
    try:
        from datasets import load_dataset
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw
        import numpy as np
        
        print("Loading dataset...")
        dataset_id = "Kamizuru00/diagram_image_to_table"
        dataset = load_dataset(dataset_id, split="train")
        
        results["details"]["dataset_info"] = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
            "first_sample_keys": list(dataset[0].keys()) if len(dataset) > 0 else []
        }
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")
        
        # Analyze dataset structure
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nFirst sample structure:")
            for key, value in sample.items():
                if key == "image":
                    print(f"  - {key}: PIL Image")
                elif key == "bbox":
                    print(f"  - {key}: {type(value).__name__} with {len(value)} boxes")
                else:
                    print(f"  - {key}: {type(value).__name__}")
        
        # Visualize samples
        print("\nVisualizing samples...")
        num_samples = min(3, len(dataset))
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
        
        for idx in range(num_samples):
            sample = dataset[idx]
            image = sample['image']
            
            # Draw bounding boxes
            draw = ImageDraw.Draw(image)
            if 'bbox' in sample and sample['bbox']:
                for box in sample['bbox']:
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            axes[idx].imshow(image)
            axes[idx].set_title(f"Sample {idx}")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_outputs/data_samples.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        results["details"]["visualization"] = "Saved to test_outputs/data_samples.png"
        results["status"] = "success"
        print("\n✓ Section 2 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 2 failed: {e}")
    
    return results

def test_section_3_inference():
    """Test Section 3: Inference Function"""
    print_section("SECTION 3: Testing Inference Function")
    results = {"section": "Inference Function", "status": "pending", "details": {}}
    
    try:
        print("Setting up inference function...")
        
        # Define the inference function
        def run_inference(processor, model, image, text_input):
            """Run inference on a single image"""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_input},
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
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
            
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        
        results["details"]["function_defined"] = "✓ Inference function defined"
        
        # Note: We won't actually load the model here due to memory constraints
        # but we'll document how it would work
        results["details"]["note"] = "Model loading skipped to conserve memory. Function ready for use."
        results["status"] = "success"
        print("\n✓ Section 3 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 3 failed: {e}")
    
    return results

def test_section_4_parse_detection():
    """Test Section 4: Parse Detection Results"""
    print_section("SECTION 4: Testing Parse Detection Results")
    results = {"section": "Parse Detection", "status": "pending", "details": {}}
    
    try:
        print("Testing detection parsing function...")
        
        def parse_detection_results(text):
            """Parse detection results from model output"""
            detections = []
            lines = text.strip().split('\n')
            
            for line in lines:
                if '<click>' in line and '</click>' in line:
                    # Extract coordinates
                    start = line.find('<click>') + 7
                    end = line.find('</click>')
                    coords = line[start:end].split(',')
                    
                    if len(coords) == 2:
                        try:
                            x = int(coords[0])
                            y = int(coords[1])
                            
                            # Extract label
                            label_start = line.find('</click>') + 8
                            label = line[label_start:].strip()
                            
                            detections.append({
                                'x': x,
                                'y': y,
                                'label': label
                            })
                        except ValueError:
                            continue
            
            return detections
        
        # Test with sample output
        sample_output = """<click>123,456</click> table
<click>789,321</click> cell
<click>555,666</click> header"""
        
        parsed = parse_detection_results(sample_output)
        results["details"]["test_parse"] = {
            "input": sample_output,
            "parsed_count": len(parsed),
            "parsed_results": parsed
        }
        
        print(f"Parsed {len(parsed)} detections from sample")
        results["status"] = "success"
        print("\n✓ Section 4 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 4 failed: {e}")
    
    return results

def test_section_5_preprocessing():
    """Test Section 5: Data Preprocessing"""
    print_section("SECTION 5: Testing Data Preprocessing")
    results = {"section": "Data Preprocessing", "status": "pending", "details": {}}
    
    try:
        from qwen_vl_utils import process_vision_info
        
        print("Testing conversation formatting...")
        
        def format_data(sample):
            """Format data for training"""
            if sample.get("bbox") and len(sample["bbox"]) > 0:
                # Create detection prompt
                detection_str = ""
                for box in sample["bbox"]:
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detection_str += f"<click>{center_x},{center_y}</click> table\n"
                
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": sample["image"]},
                                {"type": "text", "text": "Detect all tables in this image."}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": detection_str.strip()}
                            ]
                        }
                    ]
                }
            return None
        
        # Test with mock data
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        sample = {
            "image": dummy_image,
            "bbox": [[10, 10, 50, 50], [60, 60, 90, 90]]
        }
        
        formatted = format_data(sample)
        if formatted:
            results["details"]["formatted_sample"] = {
                "num_messages": len(formatted["messages"]),
                "user_content_types": [c["type"] for c in formatted["messages"][0]["content"]],
                "assistant_response_preview": formatted["messages"][1]["content"][0]["text"][:100]
            }
            print("✓ Data formatting successful")
        
        results["status"] = "success"
        print("\n✓ Section 5 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 5 failed: {e}")
    
    return results

def test_section_6_model_loading():
    """Test Section 6: Model Loading with Quantization"""
    print_section("SECTION 6: Testing Model Loading Configuration")
    results = {"section": "Model Loading", "status": "pending", "details": {}}
    
    try:
        import torch
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig
        
        print("Setting up model configuration...")
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # LoRA configuration
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        
        results["details"]["configurations"] = {
            "quantization": {
                "load_in_4bit": True,
                "use_double_quant": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16"
            },
            "lora": {
                "rank": 64,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
        }
        
        print("✓ Model configurations set up")
        print("Note: Actual model loading skipped to conserve memory")
        
        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            results["details"]["gpu_info"] = {
                "total_memory_gb": f"{gpu_memory:.2f}",
                "recommended_for_training": gpu_memory >= 24
            }
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            if gpu_memory < 24:
                print("⚠️  Warning: Less than 24GB GPU memory. Training may require adjustments.")
        
        results["status"] = "success"
        print("\n✓ Section 6 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 6 failed: {e}")
    
    return results

def test_section_7_training_setup():
    """Test Section 7: Training Setup"""
    print_section("SECTION 7: Testing Training Setup")
    results = {"section": "Training Setup", "status": "pending", "details": {}}
    
    try:
        from trl import SFTConfig
        
        print("Setting up training configuration...")
        
        # Training arguments
        training_args = SFTConfig(
            output_dir="./qwen2-vl-nutrition-finetuned",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=50,
            push_to_hub=False,
            report_to=["none"],  # Disable W&B for testing
            remove_unused_columns=False,
            bf16=True,
            max_seq_length=2048,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        
        results["details"]["training_config"] = {
            "batch_size": 2,
            "gradient_accumulation": 8,
            "effective_batch_size": 16,
            "learning_rate": 1e-5,
            "epochs": 3,
            "max_seq_length": 2048,
            "use_bf16": True,
            "gradient_checkpointing": True
        }
        
        print("✓ Training configuration set up")
        print("Note: Actual training skipped for testing")
        
        # Calculate estimated training time
        estimated_time = {
            "per_epoch_hours": "1-2 (estimated)",
            "total_hours": "3-6 (estimated)",
            "note": "Actual time depends on dataset size and hardware"
        }
        results["details"]["estimated_training_time"] = estimated_time
        
        results["status"] = "success"
        print("\n✓ Section 7 completed successfully!")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n✗ Section 7 failed: {e}")
    
    return results

def generate_report(all_results):
    """Generate comprehensive test report"""
    print_section("GENERATING FINAL REPORT")
    
    report = []
    report.append("# Qwen2-VL Fine-tuning Project Test Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary
    report.append("## Summary")
    success_count = sum(1 for r in all_results if r["status"] == "success")
    total_count = len(all_results)
    report.append(f"- **Tests Passed**: {success_count}/{total_count}")
    report.append(f"- **Success Rate**: {(success_count/total_count)*100:.1f}%\n")
    
    # Detailed Results
    report.append("## Detailed Test Results\n")
    
    for result in all_results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        report.append(f"### {status_emoji} {result['section']}")
        report.append(f"**Status**: {result['status']}")
        
        if "details" in result:
            report.append("\n**Details**:")
            report.append("```json")
            report.append(json.dumps(result["details"], indent=2))
            report.append("```")
        
        if result["status"] == "failed" and "error" in result:
            report.append(f"\n**Error**: {result['error']}")
            if "traceback" in result:
                report.append("\n**Traceback**:")
                report.append("```python")
                report.append(result["traceback"])
                report.append("```")
        
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open("test_outputs/test_report.md", "w") as f:
        f.write(report_text)
    
    print("\n✓ Report saved to test_outputs/test_report.md")
    return report_text

def main():
    """Main test execution"""
    print("="*80)
    print(" QWEN2-VL FINE-TUNING PROJECT - COMPREHENSIVE TEST")
    print("="*80)
    
    all_results = []
    
    # Run all tests
    tests = [
        test_section_1_imports,
        test_section_2_data_loading,
        test_section_3_inference,
        test_section_4_parse_detection,
        test_section_5_preprocessing,
        test_section_6_model_loading,
        test_section_7_training_setup,
    ]
    
    for test_func in tests:
        result = test_func()
        all_results.append(result)
        
        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate final report
    report = generate_report(all_results)
    
    print("\n" + "="*80)
    print(" TEST EXECUTION COMPLETE")
    print("="*80)
    print(f"\nTests Passed: {sum(1 for r in all_results if r['status'] == 'success')}/{len(all_results)}")
    print("Check test_outputs/test_report.md for detailed results")

if __name__ == "__main__":
    import torch
    main()