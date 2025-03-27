from utils.utils import grayscale, get_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_thres_image(file_path, weak_th = 50, strong_th = 150, color=(0,0,0)):
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
    if not weak_th:weak_th = mag_max * 0.05
    if not strong_th:strong_th = mag_max * 0.3
      
    # getting the dimensions of the input image   
    height, width = image.shape 
    # Looping through every pixel of the grayscale  
    # image 
    for i_x in range(width): 
        for i_y in range(height): 
            grad_ang = ang[i_y, i_x] 
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
               
            # selecting the neighbours of the target pixel 
            # according to the gradient direction 
            # In the x axis direction 
            if grad_ang<= 22.5: 
                neighb_1_x, neighb_1_y = i_x-1, i_y 
                neighb_2_x, neighb_2_y = i_x + 1, i_y 
              
            # top right (diagonal-1) direction 
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction 
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagonal-2) direction 
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle 
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
                neighb_1_x, neighb_1_y = i_x-1, i_y 
                neighb_2_x, neighb_2_y = i_x + 1, i_y 
               
            # Non-maximum suppression step 
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
                    mag[i_y, i_x]= 0
    
    # putting the calculated 
    image = cv2.imread(file_path)
    for i_x in range(width): 
        for i_y in range(height): 
            if mag[i_y, i_x] > 40:
                # make the lines green to stand out
                image[i_y, i_x] = (0, 255, 0)
    return image
    #return mag

if __name__ == '__main__':
    pic = canny_thres_image('data/V0/1.jpg')
    plt.figure() 
    f, plots = plt.subplots(2, 1)  
    plots[0].imshow(cv2.imread('data/V0/1.jpg')) 
    plots[1].imshow(pic) 
    plt.show()