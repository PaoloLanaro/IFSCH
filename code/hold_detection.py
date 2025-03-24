from canny import canny_thres_image, get_image
from otsu import otsu_thresh_image


"""
    Image is a file path
"""
def detect_hold(image):
    canny = canny_thres_image(image)
    otsu_image = get_image(image)
    otsu = otsu_thresh_image(otsu_image)
    
    print(otsu.shape)
    print(canny.shape)
    height, width = otsu.shape[:2]
    for i_x in range(width): 
        for i_y in range(height): 
            # compare ostu[i_y, i_x] with canny[i_y, i_x]

if __name__ == '__main__':
    detect_hold('data/V0/1.jpg')