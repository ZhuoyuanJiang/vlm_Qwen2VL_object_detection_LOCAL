import base64
import requests
from datasets import load_dataset

# Load a test image from the dataset
print("Loading test image from HuggingFace dataset...")
ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")
ds[0]['image'].save('/tmp/test_nutrition.jpg')
print("Saved to /tmp/test_nutrition.jpg")

# Encode image to base64
with open('/tmp/test_nutrition.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Send request to vLLM
print("\nSending request to vLLM...")
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen2vl-nutrition",
        "messages": [
            {"role": "system", "content": "You are a Vision Language Model specialized in interpreting visual data from product images. Your task is to analyze the provided product images and detect the nutrition tables in a certain format. Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Detect the bounding box coordinates for the nutrition facts table in this image."}
            ]}
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "skip_special_tokens": False  # Include special tokens in output!
    },
    timeout=60
)

# Show result
result = response.json()
print("\n" + "=" * 50)
print("Model Response:")
print("=" * 50)
print(result['choices'][0]['message']['content'])
