import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import cv2
import json
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Import the UNet model
from unn import UNet, ClimbingWallDataset

def load_model(model_path, device):
    """Load trained model"""
    model = UNet(n_channels=3, n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_mask(model, image_path, device, size=(256, 256)):
    """Predict segmentation mask for a single image"""
    # Transforms
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        output = F.softmax(output, dim=1)
        predicted_mask = output.argmax(1).squeeze().cpu().numpy()

    # Resize mask back to original image size
    predicted_mask = cv2.resize(
        predicted_mask.astype(np.uint8), 
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )

    return image, predicted_mask

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

def evaluate_model(model, test_images_dir, test_annotations_dir, device, output_dir=None):
    """Evaluate model performance on test set"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all test images
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Performance metrics
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    # Process each test image
    for i, img_file in enumerate(image_files):
        print(f"Processing test image {i+1}/{len(image_files)}: {img_file}")

        # Image path
        img_path = os.path.join(test_images_dir, img_file)

        # Predict mask
        image, pred_mask = predict_mask(model, img_path, device)

        # Get ground truth mask if available
        json_file = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(test_annotations_dir, json_file)

        if os.path.exists(json_path):
            gt_mask = load_ground_truth_mask(json_path, image.width, image.height)

            # Calculate metrics
            pred_flat = pred_mask.flatten()
            gt_flat = gt_mask.flatten()

            precision, recall, f1, _ = precision_recall_fscore_support(
                gt_flat, pred_flat, average='binary', zero_division=0)
            accuracy = accuracy_score(gt_flat, pred_flat)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)

            # Visualize and save results
            if output_dir:
                out_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_result.png")
                visualize_results(image, pred_mask, gt_mask, save_path=out_path)
            else:
                visualize_results(image, pred_mask, gt_mask)

            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        else:
            # No ground truth available, just visualize prediction
            if output_dir:
                out_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_result.png")
                visualize_results(image, pred_mask, save_path=out_path)
            else:
                visualize_results(image, pred_mask)

    # Print overall metrics
    if precisions:
        print("\nOverall Test Set Metrics:")
        print(f"Average Precision: {np.mean(precisions):.4f}")
        print(f"Average Recall: {np.mean(recalls):.4f}")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f}")

def predict_single_image(model, image_path, device, output_dir=None):
    """Predict and visualize segmentation for a single image"""
    image, pred_mask = predict_mask(model, image_path, device)

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.basename(image_path)
        out_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_result.png")
        visualize_results(image, pred_mask, save_path=out_path)
        print(f"Result saved to {out_path}")
    else:
        visualize_results(image, pred_mask)

def main():
    parser = argparse.ArgumentParser(description='Test climbing wall segmentation model')

    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'predict'], 
                        help='Evaluation mode: evaluate test set or predict single image')
    parser.add_argument('--test_images', default='../data/processed/test/images/', type=str, help='Path to test images directory')
    parser.add_argument('--test_annotations', default='../data/processed/test/annotations/', type=str, help='Path to test annotations directory')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--output', type=str, help='Output directory for results')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device)
    print(f"Model loaded from {args.model}")

    if args.mode == 'evaluate':
        if not args.test_images or not args.test_annotations:
            parser.error("--test_images and --test_annotations required for evaluation mode")

        evaluate_model(model, args.test_images, args.test_annotations, device, args.output)

    elif args.mode == 'predict':
        if not args.image:
            parser.error("--image required for prediction mode")

        predict_single_image(model, args.image, device, args.output)

if __name__ == "__main__":
    main()

