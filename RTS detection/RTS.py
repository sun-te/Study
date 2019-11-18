
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

R_THRESHOLD = 1

def threePointsToCircle(points_x, points_y):
    """
    https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
    compute a circle by three points
    return center of the circle (h,k) and its radius
    """
    x1, x2, x3 = points_x
    y1, y2, y3 = points_y
    x12 = x1 - x2  
    x13 = x1 - x3  
    y12 = y1 - y2  
    y13 = y1 - y3  
    y31 = y3 - y1  
    y21 = y2 - y1  
    x31 = x3 - x1  
    x21 = x2 - x1  
    # x1^2 - x3^2  
    sx13 = pow(x1, 2) - pow(x3, 2)  
    # y1^2 - y3^2  
    sy13 = pow(y1, 2) - pow(y3, 2)  
    sx21 = pow(x2, 2) - pow(x1, 2)  
    sy21 = pow(y2, 2) - pow(y1, 2)  
    f = (((sx13) * (x12) + (sy13) * 
          (x12) + (sx21) * (x13) + 
          (sy21) * (x13)) // (2 * 
          ((y31) * (x12) - (y21) * (x13))))         
    g = (((sx13) * (y12) + (sy13) * (y12) + 
          (sx21) * (y13) + (sy21) * (y13)) // 
          (2 * ((x31) * (y12) - (x21) * (y13))))  
    c = (-pow(x1, 2) - pow(y1, 2) - 
         2 * g * x1 - 2 * f * y1)  
    h = -g  
    k = -f  
    sqr_of_r = h * h + k * k - c
    r = np.sqrt(sqr_of_r)
    return (h, k ,r)

def numPointsOnCircle(edge, col, row, r, threshold_r = R_THRESHOLD):
    """
    Count with a threshold delta_r, how many edge points on a circle centered 
    at (row, col)
    """
    inner_r = r - threshold_r
    outer_r = r + threshold_r
    h, w = edge.shape
    ans = 0
    # h: row, w:col
    # i: row, j:col
    # y: row, x:col
    for j in range(max(0, int(col-outer_r)), min(w, int(col+outer_r))):  # row
        for i in range(max(0, int(row-outer_r)), min(h, int(row+outer_r))): # col
            if edge[i][j] > 0:
                dis = np.sqrt((j-col)**2 + (i-row)**2)
                if dis>=inner_r and dis<=outer_r:
                    ans += 1
    return ans

def circleOfInterest_random(edge, n_sample, max_r, min_r, r_threshold = R_THRESHOLD):
    """
    RANSAC detection
    """
    circle_dict = defaultdict(int)
    x, y = np.where(edge>0)
    num_points = len(x)
    print("number of points of interest {}".format(num_points))
    for i in tqdm(range(n_sample)):
        indexes = random.sample(range(0, num_points), k=3)
        points_x, points_y = x[indexes], y[indexes]
        h, k, r = threePointsToCircle(points_x, points_y)
        # h : row, k: col
        if r <= max_r and r>=min_r:
#            print("111111")
#            print(points_x, points_y)
            num_points_on_circle = numPointsOnCircle(edge,  col=k , row=h, r=r, 
                                                     threshold_r=r_threshold)
          
            if num_points_on_circle > 0:
                circle_dict[(int(h), int(k), round(r))] += num_points_on_circle/r   
    return circle_dict

def circleOfInterest_center_oriented(edge, max_r, min_r = 0,
                                     r_threshold = R_THRESHOLD):
    """
    Center oriented detection
    """
    h, w = edge.shape
#    tt()
    stock_circle = np.zeros((h, w, int(max_r)))
    x, y = np.where(edge>0)
    for index_points in tqdm(range(len(x))):
        p_x, p_y = x[index_points], y[index_points]
        
        c_x = np.array(range(max(0, p_x-max_r), min(p_x+max_r, h)))
        c_y = np.array(range(max(0, p_y-max_r), min(p_y+max_r, w)))
        
        dev_x = (c_x - p_x)**2
        dev_y = (c_y - p_y)**2
        x_mesh, y_mesh  = np.meshgrid(dev_x, dev_y)
        Z = np.sqrt(x_mesh + y_mesh)
        Z = np.maximum(Z, min_r)
        for i in range(len(c_x)):
            for j in range(len(c_y)):
                r = Z[j, i]
                if r> min_r and r<max_r-0.5:
                    stock_circle[c_x[i],c_y[j], int(round(r))] += 1/r
#        for c_x in range(max(0,p_x-max_r), min(p_x+max_r, h)):
#            for c_y in range(max(0, p_y-max_r), min(p_y+max_r, w)):
#                r = round(np.sqrt((c_x-p_x)**2 + (c_y-p_y)**2))
#                if r >min_r and r<max_r:
#                    stock_circle[c_x, c_y, r] += 1/r
    return stock_circle
   
def gray2RGB(gray):
    """
    Convert gray 2 a simple RGB three layer image
    """
    h, w = gray.shape
    rgb = np.zeros((h, w, 3))
    new_gray = (0.8 * gray).astype(int)
    rgb[..., 0] = new_gray
    rgb[..., 1] = new_gray
    rgb[..., 2] = new_gray
    rgb = rgb.astype(int)
    return rgb

def addCircle(im, row, col, r, delta_r=R_THRESHOLD, color=(255, 0, 0)):
    """
    Add a circle to the image
    """
    inner_r = r - delta_r
    outer_r = r + delta_r
    h, w = edge.shape
    # h: row, w: col
    # x: col, y: row
    for j in range(max(0, int(col-outer_r)), min(w, int(col+outer_r))):  # col
        for i in range(max(0, int(row-outer_r)), min(h, int(row+outer_r))): # row
            dis = np.sqrt((j-col)**2 + (row-i)**2)
            if dis>=inner_r and dis<=outer_r:
                im[i, j] = color
    return 

#%%

if __name__ == "__main__":
    image_names = ["RTS01org.jpg", "RTS02.jpg", "RTS03.jpg", "RTS04.jpg"]
    image_name = image_names[1]
    image_name = "sample003.jpg"
    im = cv2.imread(image_name)
    im[...,[0,2]] = im[...,[2,0]]
    image = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    plt.imshow(im) 
    plt.show()

    """
    低于阈值1的像素点会被认为不是边缘；
    高于阈值2的像素点会被认为是边缘；
    在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
    """
    #for image 1
    edge = cv2.Canny(im, threshold1 = 100, threshold2 = 250)
    #edge = cv2.Canny(im, threshold1 = 100, threshold2 = 450)
    plt.imshow(edge)
    plt.show()
    #%%
    print("Number of points of interest: {}".format(np.sum(edge>0)))
    
    """
    True: run detection with RANSAC
    False: run with center oriented detecition method
    """
    random_sample = True
    
    if random_sample:
        circle_dict = circleOfInterest_random(edge, 100, max_r=150, min_r=10)
        sorted_circle = sorted(circle_dict.items(), key=lambda x: x[1], reverse=True)
        circles = [c[0] for c in sorted_circle]
    else:
        circle_map = circleOfInterest_center_oriented(edge, max_r=50, min_r=10,
                                                       r_threshold=R_THRESHOLD)
        k_circles = 20
        circle_dict = {}
        votes = [0] * k_circles
        circles = [None] * k_circles
        shape = circle_map.shape
        for i in tqdm(range(shape[0])):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    min_votes = min(votes)
                    current_v = circle_map[i,j,k]
                    if current_v>min_votes:
                        bad_index = votes.index(min_votes)
                        votes[bad_index] = current_v
                        circles[bad_index] = (i,j,k)

        index_order = np.argsort(votes)[::-1]
        circles = np.array(circles)[index_order]

    res_image = gray2RGB(edge)
    #res_image = image.copy()
    k = 6
    #%%
    
    for c in circles[:1]:
        addCircle(res_image, col=c[1], row=c[0], r=c[2])
    fig = plt.figure()
    plt.imshow(res_image)
    plt.show()
    # Convert BGR -  RGB
    res_image[...,[0,2]] = res_image[...,[2,0]]
    cv2.imwrite("R_{}".format(image_name),res_image)
    

    plt.imshow(edge)
