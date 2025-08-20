# Fine-Tuning Qwen2-VL-7B for Nutrition Table Detection

This project implements fine-tuning of the Qwen2-VL-7B vision-language model for detecting nutrition tables in product images using QLoRA (Quantized Low-Rank Adaptation).

## 🎯 Project Overview

The goal is to fine-tune Qwen2-VL-7B to accurately detect and localize nutrition tables in product images from the OpenFoodFacts dataset. The model learns to:
- Identify nutrition tables in product images
- Output precise bounding box coordinates
- Handle various image sizes and orientations

## 📁 Project Structure

```
vlm_Qwen2VL_object_detection/
├── fine_tuning_vlm_for_object_detection_trl.ipynb  # Main notebook
├── fine_tuning_vlm_for_object_detection_trl.py     # Python script (synced with notebook)
├── requirements.txt                                  # Dependencies
├── README.md                                         # This file
└── WORKFLOW_GUIDE.md                                # Development workflow guide
```

## 🚀 Setup

### 1. Create Conda Environment

```bash
conda create -n vlm_Qwen2VL_object_detection python=3.10 -y
conda activate vlm_Qwen2VL_object_detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"  # For model access and upload
```

## 💻 Hardware Requirements

- **Recommended**: 2x NVIDIA A6000 (48GB) or 1x A100 (40GB/80GB)
- **Minimum**: 1x GPU with 24GB+ VRAM (may require smaller batch sizes)
- **Flash Attention**: Requires Ampere architecture (A100, A6000) or newer

## 📊 Dataset

The project uses the [OpenFoodFacts Nutrition Table Detection](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection) dataset:
- Training samples: ~1000
- Validation samples: ~100
- Images contain product photos with nutrition tables
- Bounding boxes are normalized to [0, 1]

## 🔧 Training Configuration

### QLoRA Settings
- **Rank (r)**: 64
- **Alpha**: 128
- **Target modules**: Attention and MLP layers
- **Quantization**: 4-bit NF4

### Training Hyperparameters
- **Learning rate**: 2e-4
- **Batch size**: 2 per device
- **Gradient accumulation**: 8 steps
- **Epochs**: 3
- **Mixed precision**: BF16

## 📈 Expected Results

After fine-tuning, the model shows significant improvement:
- **Mean IoU**: ~0.65-0.75 (vs ~0.2-0.3 for base model)
- **Detection Rate**: >90% (vs ~50% for base model)
- **IoU > 0.5**: >70% of samples

## 🏃 Running the Code

### Option 1: Jupyter Notebook
```bash
jupyter notebook fine_tuning_vlm_for_object_detection_trl.ipynb
```

### Option 2: Python Script
```bash
python fine_tuning_vlm_for_object_detection_trl.py
```

### Syncing Files (Jupytext)
The notebook and Python script are kept in sync using Jupytext:
```bash
jupytext --sync fine_tuning_vlm_for_object_detection_trl.py
```

## 📝 Key Implementation Details

1. **Data Preprocessing**: Converts dataset to conversation format with system, user, and assistant roles
2. **Custom Collator**: Handles text-image pairs and masks non-answer tokens for loss computation
3. **Evaluation**: Uses IoU (Intersection over Union) metric for bounding box accuracy
4. **Model Merging**: LoRA weights can be merged into base model for deployment

## 🔍 Monitoring

Training progress is tracked with Weights & Biases (W&B):
- Loss curves
- Evaluation metrics
- Learning rate schedule
- GPU utilization

## 🚢 Deployment

After training, the model can be:
1. Used with LoRA adapters for efficient inference
2. Merged into base model for standalone deployment
3. Exported to ONNX/TensorRT for production use

## 📚 References

- [Qwen2-VL Paper](https://arxiv.org/pdf/2409.12191)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

## ⚠️ Notes

- The implementation focuses on single bounding box detection per image
- For multiple objects, the code can be extended to handle multiple detections
- Flash Attention requires specific GPU architectures for optimal performance

## 📄 License

This project is for educational purposes. Please refer to the respective licenses for:
- Qwen2-VL model
- OpenFoodFacts dataset
- Hugging Face libraries