import os
import json
import shutil
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

def create_directory_structure(base_dir):
    """Create the necessary directory structure for the dataset"""
    # Remove the directory if it exists so that we don't get erroneous data (if we change random_state)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"Deleted existing directory and its contents: {base_dir}")

    # Create main directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'annotations']:

            os.makedirs(os.path.join(base_dir, split, subdir), exist_ok=True)

    print(f"Created directory structure in {base_dir}")

def vgg_json_to_mask(json_path, image_width, image_height):
    """Convert VGG Image Annotator JSON to binary mask"""
    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Read the JSON file
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
                        cv2.rectangle(mask, tuple(p1), tuple(p2), 1, -1)

        except Exception as e:
            print(f"Error parsing JSON {json_path}: {e}")

    return mask

def process_dataset(input_images_dir, input_json_path, base_dir, seed, test_size=0.2, val_size=0.1):
    """Process the dataset and split it into train, validation, and test sets"""
    # Get list of all images
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images")

    # Load the merged JSON file
    try:
        with open(input_json_path, 'r') as f:
            merged_data = json.load(f)

        # Check if it's a VGG Image Annotator format
        if '_via_img_metadata' in merged_data:
            # Get all annotated image filenames from the JSON
            annotated_filenames = set()
            for key, img_data in merged_data['_via_img_metadata'].items():
                filename = img_data.get('filename', '')
                if filename:
                    annotated_filenames.add(filename)

            # Filter images that have annotations
            valid_image_files = [img for img in image_files if img in annotated_filenames]

            print(f"Found {len(valid_image_files)} images with annotations")

            if len(valid_image_files) == 0:
                print("Error: No images with corresponding annotations found!")
                return

            # Split data into train, validate, and test
            train_files, rest_files = train_test_split(valid_image_files, train_size=0.8, random_state=seed)
            val_files, test_files = train_test_split(rest_files, test_size=0.5, random_state=seed)
            print(f"Split dataset: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images")

            # Process files for each split
            process_files('train', train_files, input_images_dir, merged_data, base_dir)
            process_files('val', val_files, input_images_dir, merged_data, base_dir)
            process_files('test', test_files, input_images_dir, merged_data, base_dir)
        else:
            print("Error: The JSON file does not appear to be in VGG Image Annotator format!")

    except Exception as e:
        print(f"Error loading JSON file: {e}")

def process_files(split, file_list, input_images_dir, merged_data, base_dir):
    """Process and copy files for a specific split (train/val/test)"""
    for image_file in file_list:
        # Image paths
        src_image_path = os.path.join(input_images_dir, image_file)
        dst_image_path = os.path.join(base_dir, split, 'images', image_file)

        # Copy image
        shutil.copy2(src_image_path, dst_image_path)

        # Create a separate JSON file for this image from the merged data
        if '_via_img_metadata' in merged_data:
            # Find this image in the metadata
            image_data = None
            for key, img_data in merged_data['_via_img_metadata'].items():
                if img_data.get('filename') == image_file:
                    image_data = img_data
                    break

            if image_data:
                # Create a new VGG JSON structure for just this image
                single_image_json = {
                    '_via_settings': merged_data.get('_via_settings', {}),
                    '_via_attrs': merged_data.get('_via_attrs', {}),
                    '_via_img_metadata': {
                        key: image_data
                    }
                }

                # Write to a new JSON file
                dst_json_path = os.path.join(base_dir, split, 'annotations', os.path.splitext(image_file)[0] + '.json')
                with open(dst_json_path, 'w') as f:
                    json.dump(single_image_json, f)
            else:
                print(f"Warning: Could not find annotation for {image_file} in the merged JSON")


def visualize_sample_data(base_dir, num_samples=3):
    """Visualize some sample images and their masks from the training set"""
    train_images_dir = os.path.join(base_dir, 'train', 'images')
    train_annot_dir = os.path.join(base_dir, 'train', 'annotations')

    # Get a few sample images
    image_files = os.listdir(train_images_dir)
    if len(image_files) == 0:
        print("No images found in the training directory!")
        return

    sample_files = image_files[:min(num_samples, len(image_files))]

    for image_file in sample_files:
        # Get image and annotation paths
        image_path = os.path.join(train_images_dir, image_file)
        json_path = os.path.join(train_annot_dir, os.path.splitext(image_file)[0] + '.json')

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Generate mask from JSON
        if os.path.exists(json_path):
            mask = vgg_json_to_mask(json_path, image.width, image.height)

            # Create a colored overlay
            colored_mask = np.zeros_like(image_np)
            colored_mask[mask == 1] = [255, 0, 0]  # Red for climbing holds

            # Blend image with mask
            alpha = 0.5
            blended = cv2.addWeighted(image_np, 1, colored_mask, alpha, 0)

            # Display
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
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
            plt.show()
        else:
            print(f"No annotation found for {image_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare climbing wall segmentation dataset')
    parser.add_argument('--images', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--annotation', type=str, required=True, help='Path to a VGG IA JSON annotation file')
    # Default to the data directory at project root
    parser.add_argument('--output', type=str, default='../data/processed', help='Output base directory for processed dataset')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample data after processing')
    parser.add_argument('--seed', type=int, default=43, help='Seed the splitting to a specified number')

    args = parser.parse_args()

    # Create directory structure
    create_directory_structure(args.output)

    # Process and split the dataset
    process_dataset(args.images, args.annotation, args.output, args.seed)

    if args.visualize:
        visualize_sample_data(args.output)

    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()

