import numpy as np
from PIL import Image
import os

def check_file_path(file_path):
    return os.path.isfile(file_path)

def get_image(file_path):
    if check_file_path(file_path):
        return Image.open(file_path)
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")

def convert_image_to_greyscale(image):
    return image.convert('L')

def convert_image_to_array(image):
    return np.array(image)

def flatten_image_arr(img_arr):
    return img_arr.flatten()

def otsu_thresh_image(image):
    img_arr = convert_image_to_array(image)

    img_flat = flatten_image_arr(img_arr)

    pixel_counts = np.bincount(img_flat, minlength=256)

    # get probability for each pixel count
    probabilities = pixel_counts / img_flat.shape[0]

    # cumulative probability over the pixels
    cumulative_p = np.cumsum(probabilities)
    cumulative_sum = np.cumsum(np.arange(256) * probabilities)

    total_mean = cumulative_sum[-1]

    max_var = float('-inf')
    optimal_threshold = 0

    for t in range(256):
        w0 = cumulative_p[t]
        # remove some tolerance for non divide by zero
        if w0 < 1e-10 or w0 > 1 - 1e-10:
            continue

        w1 = 1 - w0
        sum_pixel = cumulative_sum[t]
        mean0 = sum_pixel / w0
        mean1 = (total_mean - sum_pixel) / w1

        var = w0 * w1 * (mean0 - mean1) ** 2

        if var > max_var:
            max_var = var
            optimal_threshold = t

    binary_array = np.where(img_arr > optimal_threshold, 255, 0).astype(np.uint8)
    # returning just the binary array bc it allows us to compare with canny
    return binary_array
    #return Image.fromarray(binary_array)
