from PIL import Image
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



def grayscale(img):
    """
    Make sure it's a PIL image
    """
    return img.convert('L')
def check_file_path(file_path):
    return os.path.isfile(file_path)

def get_image(file_path):
    if check_file_path(file_path):
        return Image.open(file_path)
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
def visualize_results(image, mask, gt_mask=None, alpha=0.5, save_path=None):
    """Visualize the segmentation results"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Create a colored mask (red for predictions)
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [255, 0, 0]  # Red for predicted holds

    # Blend image with mask
    blended = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    if gt_mask is not None:
        # Use 4 subplots if ground truth is available
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Predicted Segmentation')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Create a colored overlay showing true positives, false positives, etc.
        comparison = np.zeros_like(image)
        comparison[(mask == 1) & (gt_mask == 1)] = [0, 255, 0]    # True positive (green)
        comparison[(mask == 1) & (gt_mask == 0)] = [255, 0, 0]    # False positive (red)

        blended_comparison = cv2.addWeighted(image, 0.7, comparison, 0.3, 0)

        plt.subplot(2, 2, 4)
        plt.imshow(blended_comparison)
        plt.title('Comparison (Green: TP, Red: FP)')
        plt.axis('off')
    else:
        # Use 3 subplots if no ground truth is available
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(blended)
        plt.title('Segmentation Overlay')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def load_ground_truth_mask(json_path, image_width, image_height):
    """Load ground truth mask from VGG IA JSON format"""
    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Read the JSON file
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)

                # Handle VGG Image Annotator format
                if '_via_img_metadata' in data:
                    # Get the first key in _via_img_metadata
                    img_key = next(iter(data['_via_img_metadata']))
                    img_data = data['_via_img_metadata'][img_key]

                    # Process all regions
                    for region in img_data.get('regions', []):
                        shape_attrs = region.get('shape_attributes', {})

                        if shape_attrs.get('name') == 'polygon':
                            # Get polygon points
                            x_points = shape_attrs.get('all_points_x', [])
                            y_points = shape_attrs.get('all_points_y', [])

                            # Create points array for CV2
                            if len(x_points) == len(y_points) and len(x_points) > 2:
                                points = np.array(list(zip(x_points, y_points)), dtype=np.int32)
                                # Fill polygon
                                cv2.fillPoly(mask, [points], 1)

                        elif shape_attrs.get('name') == 'rect':
                            # Get rectangle coordinates
                            x = shape_attrs.get('x', 0)
                            y = shape_attrs.get('y', 0)
                            width = shape_attrs.get('width', 0)
                            height = shape_attrs.get('height', 0)

                            # Fill rectangle
                            cv2.rectangle(mask, (x, y), (x + width, y + height), 1, -1)

                # Handle simpler format
                elif 'shapes' in data:
                    for shape in data['shapes']:
                        if shape.get('shape_type') == 'polygon':
                            points = np.array(shape['points'], dtype=np.int32)
                            cv2.fillPoly(mask, [points], 1)
                        elif shape.get('shape_type') == 'rectangle':
                            p1, p2 = shape['points']
                            cv2.rectangle(mask, (tuple(p1)), (tuple(p2)), 1, -1)

            except Exception as e:
                print(f"Error parsing JSON {json_path}: {e}")

    return mask