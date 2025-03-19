from PIL import Image

def grayscale(img):
    """
    Make sure it's a PIL image
    """
    return img.convert('L')