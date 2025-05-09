from canny import canny_thres_image, get_image, canny_thres_color
from otsu import otsu_thresh_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
"""
    Image is a file path
"""
def detect_hold(image):
    # TODO GIVE AN INPUT INTO CANNY AND OTSU TO LOOK FOR A SPECIFIC COLOR SO THAT WE CAN INPUT 
    # AN IMAGE WITH A COLOR AND HAVE THE DETECTION DETECT A CLIMB WITH THAT COLOR

    print(f'Segmenting image {image}')

    canny_start_time = datetime.datetime.now()
    canny = canny_thres_color(image, color='blue')
    canny_end_time = datetime.datetime.now()
    print(f'Canny runtime {canny_end_time-canny_start_time}')
    otsu_image = get_image(image)
    otsu_start_time = datetime.datetime.now()
    otsu = otsu_thresh_image(otsu_image)
    otsu_end_time = datetime.datetime.now()
    print(f'Otsu runtime {otsu_end_time-otsu_start_time}')
    
    height, width = otsu.shape[:2]
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i_x in range(width): 
        for i_y in range(height): 
            # compare ostu[i_y, i_x] with canny[i_y, i_x]
            if np.array_equal(canny[i_y, i_x], [0, 255, 0]):
                image[i_y, i_x] = (0, 255, 0)
            # elif np.array_equal(otsu[i_y, i_x], [0, 0, 0]):
            #     image[i_y, i_x] = (255, 0, 0)
    return image
if __name__ == '__main__':
    f, plots = plt.subplots(8, 4, figsize=(20,10))  
    for i in range(8):
        plt.clf()
        plt.subplot(1, 2, 1)

        image_path = os.path.join('data', f'V{i}', '1.jpg')
        print("Trying to read:", image_path)
        print("Full path:", os.path.abspath(image_path))

        img = cv2.imread(image_path)
        if img is None:
            print("Failed to load image.")
        else:
            print("Image loaded successfully.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            plt.imshow(img)
        plt.imshow(cv2.cvtColor(cv2.imread(f'data/V{i}/1.jpg'), cv2.COLOR_BGR2RGBA))
        plt.title(f'Original {i}')

        plt.subplot(1, 2, 2)
        plt.imshow(detect_hold(f'data/V{i}/1.jpg'))  # Assuming pic_list contains 10 processed images
        plt.title(f'Processed {i}')
        plt.show()
        #plt.savefig(f'data/blue/V{i}_blue.jpg')
    