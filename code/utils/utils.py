from PIL import Image
import os

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