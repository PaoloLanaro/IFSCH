import matplotlib.pyplot as plt
from PIL import Image
from otsu import *

def test_otsu_implementation(file_path):
    # load image
    try:
        # test file loading
        print("Testing file loading")
        img = get_image(file_path)
        print("File loaded")

        # test grayscale
        print("\nTesting grayscale conversion")
        grey_img = convert_image_to_greyscale(img)
        grey_img_array = convert_image_to_array(grey_img)
        print(f"Grayscale conversion successful. Shape: {grey_img_array.shape}, Mode: {grey_img.mode}")

        # test Otsu thresh
        print("\nTesting Otsu thresholding")
        binary_img = otsu_thresh_image(grey_img)
        binary_array = convert_image_to_array(binary_img)
        print(f"Thresholding complete. Unique values: {np.unique(binary_array)}")

        # visualization
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # original
        ax[0, 0].imshow(img)
        ax[0, 0].set_title('Original Image')
        ax[0, 0].axis('off')

        # grayscale image
        ax[0, 1].imshow(grey_img, cmap='gray')
        ax[0, 1].set_title('Grayscale Image')
        ax[0, 1].axis('off')
        assert grey_img.mode == 'L', "Grayscale image should be in 'L' mode"

        # binary image
        ax[1, 0].imshow(binary_img, cmap='gray')
        ax[1, 0].set_title('Otsu Thresholded Image')
        ax[1, 0].axis('off')
        assert set(np.unique(binary_array)) <= {0, 255}, "Binary image should only contain 0 and 255"

        # hist with threshold
        ax[1, 1].hist(grey_img_array.flatten(), bins=256, range=[0,256], density=True)
        threshold = np.mean(binary_array)  # Get actual threshold from binary image
        ax[1, 1].axvline(x=threshold, color='r', linestyle='--')
        ax[1, 1].set_title('Intensity Histogram with Otsu Threshold')
        ax[1, 1].set_xlabel('Pixel Intensity')
        ax[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        print("\nAll tests passed")

    except Exception as e:
        print(f"\nTest failed: {str(e)}")

if __name__ == "__main__":
    test_image_path = './data/V0/1.jpg'
    test_otsu_implementation(test_image_path)
