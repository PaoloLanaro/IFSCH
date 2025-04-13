from canny import canny_thres_color
from otsu import otsu_thresh_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from utils.utils import load_ground_truth_mask
import argparse

def detect_hold(image, color):
    """
    Calls our implementation of Canny with color input to do edge detection
    Returns a numpy array representation of the segmentation
    """

    print(f'Segmenting image {image}')

    canny_start_time = datetime.datetime.now()
    canny = canny_thres_color(image, color=color)
    canny_end_time = datetime.datetime.now()
    print(f'Canny runtime {canny_end_time-canny_start_time}')
    
    height, width = canny.shape[:2]
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i_x in range(width): 
        for i_y in range(height): 
            # check if the specific pixel has a prediction from canny
            if np.array_equal(canny[i_y, i_x], [0, 255, 0]):
                image[i_y, i_x] = (0, 255, 0)
    return image

def evaluate_model(test_images_dir, test_annotations_dir, output_dir=None, show=True, colors=None):
    """Evaluate model performance on test set"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Get list of all test images
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    accuracy_by_color = {}

    # Process each image
    for i, img_file in enumerate(image_files):
        TP=0
        FP=0
        if '._' in img_file:
            continue
        print(f"Processing test image {i+1}/{len(image_files)}: {img_file}")

        # Image path
        img_path = os.path.join(test_images_dir, img_file)
        
        
        # Predict mask based on color
        for color in colors:
            overall_img = cv2.imread(img_path)
            image = detect_hold(img_path, color)
            
            # separate green outline with non green
            green_mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 255) & (image[:, :, 2] == 0)
            green_mask = green_mask.astype(np.uint8) * 255

            # find holds
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)

            #label_img = cv2.imread(img_path)
            label_img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

            min_area = 50
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < min_area:
                    # set this pixel to the background aka 0
                    # each label is a different region
                    labels[labels == label] = 0
            
            # Get ground truth mask if available
            json_file = os.path.splitext(img_file)[0] + '.json'
            json_path = os.path.join(test_annotations_dir, json_file)
            TP = 0
            FP = 0
            if os.path.exists(json_path):
                gt_mask = load_ground_truth_mask(json_path, image.shape[1], image.shape[0])
                gt_mask_rgb = cv2.cvtColor(gt_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)  # Convert GT to RGB

                for i, row in enumerate(np.unique(labels)):
                    if i == 0:
                        continue
                    print('row', row)
                    predicted_region = (labels == row).astype(np.uint8)
                    overlap = np.any((predicted_region == 1) & (gt_mask == 1))
                    if color not in accuracy_by_color:
                        accuracy_by_color[color] = {}
                        accuracy_by_color[color]['TP'] = 0
                        accuracy_by_color[color]['FP'] = 0
                    if overlap:
                        print(f"At least one GT pixel is inside the predicted region {row}.")
                        TP += 1
                        accuracy_by_color[color]['TP'] += 1
                    else:
                        print(f"No GT pixels are inside the predicted region {row}.")
                        FP += 1
                        accuracy_by_color[color]['FP'] += 1
                    set_color = np.random.randint(0, 255, size=3)  # Generate a random color

                    label_img[predicted_region == 1] = set_color  # Apply color to predicted region
                    overall_img[predicted_region == 1] = set_color
                print(f"True Positives: {TP}")
                print(f"False Positives: {FP}")
            
            display_plots(overall_img, gt_mask_rgb, label_img, show, TP, FP, color, img_file)
    print('Accuracy by color', accuracy_by_color)
    display_accuracy_by_color(accuracy_by_color=accuracy_by_color, show=show)

def predict_single_image(img_path, show, colors):
    """
    Predict and visualize edge detection for a single hold
    """
    # Predict mask
    for color in colors:
        label_img = cv2.imread(img_path)
        print('color', color)
        image = detect_hold(img_path, color)
        # separate green outline with non green
        green_mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 255) & (image[:, :, 2] == 0)
        green_mask = green_mask.astype(np.uint8) * 255

        # find holds
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)

        min_area = 50
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                # set this pixel to the background aka 0
                # each label is a different region
                labels[labels == label] = 0

        # Create a blank image for coloring
        overlay = np.zeros_like(label_img)

        # Assign a color (e.g., red) to each label
        for label in range(1, np.max(labels) + 1):
            overlay[labels == label] = [0, 255, 0]  # Red color

        # Blend overlay with original image
        combined = cv2.addWeighted(label_img, 1, overlay, 0.5, 0)

        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGBA))
        plt.title(f"Prediction with Color:{color}")
        plt.axis('off')
        if show:
            plt.show()
        else:
            filename = os.path.basename(img_path)
            print(filename)
            result = f'../data/canny_results/predict/{color}_{filename}'
            print(result)
            plt.savefig(result, dpi=300, format='jpg')
        
def display_plots(original_img, gt_img, pred_img, show, TP, FP, color, path):
    """
    Visualize three plots: Original image, GT segmentation of holds, and Predicted edge detection of holds
    """
    plt.figure(figsize=(30, 10))  # Larger figure size to fit both images

    plt.subplot(1, 3, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGBA))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)  # 1 row, 2 columns, first subplot
    plt.imshow(gt_img)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGBA))
    plt.title(f"Num TP: {TP}, Num FP:{FP}, Color:{color}")
    plt.axis('off')
    
    plt.tight_layout(pad=2)  
    plt.subplots_adjust(wspace=0.1)
    if show:
        plt.show()
    else:
        plt.savefig(f'../data/canny_results/evaluate/{color}_{path}', dpi = 300)
    plt.clf()

def display_accuracy_by_color(accuracy_by_color, show):
    # extract data
    colors = list(accuracy_by_color.keys())
    tp_values = [accuracy_by_color[color]['TP'] for color in colors]
    fp_values = [accuracy_by_color[color]['FP'] for color in colors]

    x = range(len(colors))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # TP bars
    ax.bar([i - bar_width/2 for i in x], tp_values, width=bar_width, label='TP', color='skyblue')

    # FP bars
    ax.bar([i + bar_width/2 for i in x], fp_values, width=bar_width, label='FP', color='salmon')

    # Labels and titles
    ax.set_xlabel('Color')
    ax.set_ylabel('Count')
    ax.set_title('True Positives (TP) and False Positives (FP) by Color')
    ax.set_xticks(x)
    ax.set_xticklabels(colors)
    ax.legend()

    plt.tight_layout()
    if show: 
        plt.show()
    else:
        plt.savefig('../data/canny_results/accuracy_by_color.jpg', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test climbing wall segmentation model')
    parser.add_argument('--test_images', default='../data/test/images', type=str, help='Path to test images directory')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'predict'], 
                        help='Evaluation mode: evaluate test set or predict single image')
    parser.add_argument('--test_annotations', default= '../data/test/annotations',type=str, help='Path to test annotations directory')
    parser.add_argument('--show', action='store_true', help='Show results if set; otherwise, results are saved')
    parser.add_argument('--colors', default=['red'], nargs='+', type= str, help='Specifies which colors to look for')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
   
    args = parser.parse_args()

    if args.mode == 'evaluate':
        evaluate_model(test_annotations_dir=args.test_annotations, test_images_dir=args.test_images, show=args.show, colors=args.colors)

    elif args.mode == 'predict':
        if not args.image:
            parser.error("--image required for prediction mode")

        predict_single_image(args.image, args.show, args.colors)

    #evaluate_model(test_annotations_dir='data/test/annotations', test_images_dir='data/test/images')
    

    