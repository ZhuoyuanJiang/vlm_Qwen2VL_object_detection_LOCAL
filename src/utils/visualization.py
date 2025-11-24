"""
Visualization Utilities for Bounding Box Overlays

QUICK REFERENCE:
- WHEN: After getting bbox predictions from model
- PURPOSE: Draw bounding boxes on images for visual inspection
- RUNS: On-demand for visualization (not during training)
"""

from PIL import ImageDraw, ImageFont


def visualize_bbox_on_image(image, bbox_data, normalize_coords=True):
    """
    Visualize bounding boxes on an image.

    WHEN: After parsing model output, to visualize predictions

    INPUT:
        image: PIL Image object
        bbox_data: Dict or list of dicts with 'object' and 'bbox' keys
                  - Single: {'object': 'car', 'bbox': [x1, y1, x2, y2]}
                  - Multiple: [{'object': 'car', 'bbox': [...]}, ...]
        normalize_coords: If True, bbox coords are in Qwen's [0,1000] space
                         If False, bbox coords are already in pixel coordinates

    OUTPUT:
        PIL Image: New image with bounding boxes drawn

    COORDINATE CONVERSION:
        If normalize_coords=True:
            - Input bbox: [x1, y1, x2, y2] in [0, 1000] space (from model)
            - Conversion: pixel_x = (x * image_width) / 1000
            - Output: Draws bbox in actual pixel coordinates

        If normalize_coords=False:
            - Input bbox: [x1, y1, x2, y2] already in pixels
            - No conversion needed

    VISUAL PROPERTIES:
        - Rectangle outline: 4px width
        - Colors: Cycles through ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        - Label: Object name above bbox with colored background
        - Font: DejaVuSans-Bold 20pt (fallback if not found)

    Example Usage:
        >>> from PIL import Image
        >>> from src.models.inference import parse_qwen_bbox_output
        >>>
        >>> # Single bbox from model output
        >>> image = Image.open("product.jpg")
        >>> model_output = "nutrition table(13,60),(984,989)"
        >>> bbox_data = parse_qwen_bbox_output(model_output)
        >>> img_with_bbox = visualize_bbox_on_image(image, bbox_data, normalize_coords=True)
        >>> img_with_bbox.show()
        >>>
        >>> # Multiple bboxes
        >>> bbox_data = [
        ...     {'object': 'table 1', 'bbox': [10, 20, 100, 200]},
        ...     {'object': 'table 2', 'bbox': [500, 600, 900, 900]}
        ... ]
        >>> img_with_bboxes = visualize_bbox_on_image(image, bbox_data, normalize_coords=True)

    Why This Function Exists:
        - Quick visual validation of model predictions
        - Compare ground truth vs predictions side-by-side
        - Debug coordinate conversion issues
        - Educational: shows students what model detected
    """
    # Make a copy to avoid modifying original
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    width, height = img_with_bbox.size

    # Handle None case
    if bbox_data is None:
        print("No bounding box data to visualize")
        return img_with_bbox

    # Handle single or multiple bboxes
    if isinstance(bbox_data, dict):
        bbox_data = [bbox_data]

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    for idx, data in enumerate(bbox_data):
        if data is None:
            continue

        color = colors[idx % len(colors)]
        bbox = data['bbox']
        object_name = data.get('object', 'unknown')

        # Convert coordinates if needed
        if normalize_coords:
            # Qwen uses 1000x1000 normalized space
            x1 = int(bbox[0] * width / 1000)
            y1 = int(bbox[1] * height / 1000)
            x2 = int(bbox[2] * width / 1000)
            y2 = int(bbox[3] * height / 1000)
        else:
            x1, y1, x2, y2 = bbox

        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle with thicker line
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Add label with background
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None

        label = f"{object_name}"
        if font:
            # Get text size for background
            bbox_label = draw.textbbox((x1, y1-30), label, font=font)
            # Draw background rectangle
            draw.rectangle(bbox_label, fill=color)
            # Draw text
            draw.text((x1, y1-30), label, fill='white', font=font)
        else:
            # Fallback without custom font
            draw.text((x1, y1-25), label, fill=color)

        print(f"Drew bbox in pixels for '{object_name}': [{x1}, {y1}, {x2}, {y2}]")

    return img_with_bbox


def visualize_ground_truth_bbox(image, bboxes, categories, format='openfoodfacts'):
    """
    Visualize ground truth bounding boxes from dataset.

    WHEN: For comparing ground truth vs model predictions

    INPUT:
        image: PIL Image object
        bboxes: List of bbox coordinates
        categories: List of category names (same length as bboxes)
        format: Dataset format ('openfoodfacts' or 'qwen')
                - 'openfoodfacts': [y_min, x_min, y_max, x_max] normalized [0,1]
                - 'qwen': [x_min, y_min, x_max, y_max] normalized [0,1000]

    OUTPUT:
        PIL Image: New image with ground truth bboxes drawn in green

    COORDINATE FORMATS:
        OpenFoodFacts format:
            - Order: [y_min, x_min, y_max, x_max]
            - Space: Normalized [0, 1] (multiply by image dimensions)
            - Example: [0.1, 0.2, 0.9, 0.8]

        Qwen format:
            - Order: [x_min, y_min, x_max, y_max]
            - Space: Normalized [0, 1000] (multiply by image dimensions, divide by 1000)
            - Example: [100, 200, 900, 800]

    Example Usage:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("openfoodfacts/nutrition-table-detection", split="train")
        >>> example = dataset[0]
        >>>
        >>> img_with_gt = visualize_ground_truth_bbox(
        ...     example['image'],
        ...     example['objects']['bbox'],
        ...     example['objects']['category_name'],
        ...     format='openfoodfacts'
        ... )
        >>> img_with_gt.show()
    """
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    pil_width, pil_height = img_with_bbox.size

    for bbox_idx, (bbox, category) in enumerate(zip(bboxes, categories)):
        if format == 'openfoodfacts':
            # OpenFoodFacts format: [y_min, x_min, y_max, x_max] normalized [0,1]
            y_min, x_min, y_max, x_max = bbox
            gt_x1 = int(x_min * pil_width)
            gt_y1 = int(y_min * pil_height)
            gt_x2 = int(x_max * pil_width)
            gt_y2 = int(y_max * pil_height)
        elif format == 'qwen':
            # Qwen format: [x_min, y_min, x_max, y_max] normalized [0,1000]
            x_min, y_min, x_max, y_max = bbox
            gt_x1 = int(x_min * pil_width / 1000)
            gt_y1 = int(y_min * pil_height / 1000)
            gt_x2 = int(x_max * pil_width / 1000)
            gt_y2 = int(y_max * pil_height / 1000)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'openfoodfacts' or 'qwen'")

        # Draw green rectangle for ground truth
        draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2],
                      outline='green', width=4)

        # Add label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None

        label = f"GT {bbox_idx+1}: {category}"
        if font:
            bbox_label = draw.textbbox((gt_x1, gt_y1-30), label, font=font)
            draw.rectangle(bbox_label, fill='green')
            draw.text((gt_x1, gt_y1-30), label, fill='white', font=font)
        else:
            draw.text((gt_x1, gt_y1-25), label, fill='green')

        print(f"Drew ground truth bbox {bbox_idx+1} for '{category}': [{gt_x1}, {gt_y1}, {gt_x2}, {gt_y2}]")

    return img_with_bbox
