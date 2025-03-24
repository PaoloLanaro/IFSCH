from canny import canny_thres_image, get_image
from otsu import otsu_thresh_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    Image is a file path
"""
def detect_hold(image):
    canny = canny_thres_image(image)
    otsu_image = get_image(image)
    otsu = otsu_thresh_image(otsu_image)
    
    height, width = otsu.shape[:2]
    image = cv2.imread(image)
    for i_x in range(width): 
        for i_y in range(height): 
            # compare ostu[i_y, i_x] with canny[i_y, i_x]
            # print('otsu', otsu[i_y, i_x])
            # print('canny', canny[i_y, i_x])
            if np.array_equal(canny[i_y, i_x], [0, 255, 0]):
                image[i_y, i_x] = (0, 255, 0)
            elif np.array_equal(otsu[i_y, i_x], [0, 0, 0]):
                image[i_y, i_x] = (0, 0, 0)
    return image
if __name__ == '__main__':
    pic = detect_hold('data/V0/1.jpg')
    f, plots = plt.subplots(2, 1)  
    plots[0].imshow(cv2.imread('data/V0/1.jpg'))
    plots[1].imshow(pic) 
    plt.show()