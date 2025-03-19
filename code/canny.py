from utils.utils import grayscale, get_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_thres_image(file_path, weak_th = None, strong_th = None):
    image = cv2.imread(file_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Noise reduction step 
    image = cv2.GaussianBlur(image, (5, 5), 1.4) 
       
    # Calculating the gradients 
    gx = cv2.Sobel(np.float32(image), cv2.CV_64F, 1, 0, 3) 
    gy = cv2.Sobel(np.float32(image), cv2.CV_64F, 0, 1, 3) 

    # Conversion of Cartesian coordinates to polar  
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
       
    # setting the minimum and maximum thresholds  
    # for double thresholding 
    mag_max = np.max(mag) 
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image   
    height, width = image.shape 

canny_thres_image('data/V0/1.jpg')
