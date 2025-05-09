from utils.utils import grayscale, get_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_thres_color(file_path, weak_th=50, strong_th=150, color='red'):
    print('file path', file_path)
    image = cv2.imread(file_path)
    image = cv2.medianBlur(image, 7)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_pink = np.array([170, 50, 50])
    upper_pink = np.array([180, 255, 255])
    
    lower_yellow = np.array([23, 60, 70])
    upper_yellow = np.array([35, 200, 200])
    
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    
    lower_green = np.array([36, 50, 70])
    upper_green = np.array([89, 255, 255])

    lower_purple = np.array([120, 50, 70])
    upper_purple = np.array([[158, 255, 255]])

    color_range = []
    if color == 'blue':
        color_range = (lower_blue, upper_blue)
    elif color == 'red':
        color_range = (lower_red1, upper_red1)
    elif color == 'green':
        color_range = (lower_green, upper_green)
    elif color == 'pink':
        color_range = (lower_pink, upper_pink)
    elif color == 'yellow':
        color_range = (lower_yellow, upper_yellow)
    elif color == 'purple':
        color_range = (lower_purple, upper_purple)
    print('range', color_range)
    # Create masks
    mask = cv2.inRange(hsv, color_range[0], color_range[1])
    
    # Apply mask to extract color parts
    
    result = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Noise reduction step
    gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(gray), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(gray), cv2.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar  
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag_max = np.max(mag)
    if not weak_th:
        weak_th = mag_max * 0.05
    if not strong_th:
        strong_th = mag_max * 0.3
    height, width = gray.shape 
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
    #image = cv2.imread(file_path)
    for i_x in range(width): 
        for i_y in range(height): 
            if mag[i_y, i_x] > 40:
                # make the lines green to stand out
                image[i_y, i_x] = (0, 255, 0)
    return image
    
if __name__ == '__main__':
    pic = canny_thres_color('data/test/images/2.jpg', color='red')
    plt.figure() 
    f, plots = plt.subplots(2, 1)  
    plots[0].imshow(cv2.cvtColor(cv2.imread('data/test/images/2.jpg'), cv2.COLOR_BGR2RGBA)) 
    plots[1].imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGBA)) 
    plt.show()