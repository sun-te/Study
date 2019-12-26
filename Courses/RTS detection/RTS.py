try:
    # opencv用来canny edge detection
    import cv2 
except:
    print("Please install opencv: pip install opencv-python")
try:
    # 看循环进度
    from tqdm import tqdm
except:
    print("Plase install tqdm: pip install tqdm")
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict


import os
"""
Change to your local working space
"""
os.chdir("..\\RTS detection\\")

# R threshold for ransac
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
    stock_circle = np.zeros((h, w, int(max_r+1)))
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
        for i in range(0, len(c_x), 1):
            for j in range(0, len(c_y), 1):
                r = Z[j, i]
                if r> min_r and r<max_r:
                    stock_circle[c_x[i],c_y[j], int(round(r))] += 1./int(r)
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


def maximum_suppresion(circle,  votes,threshold):
    num_circle = len(circle)
    for i in range(num_circle):
        if sum(circle[i]) == 0:
            continue
        for j in range(i+1, num_circle):
            if sum(circle[j]) == 0:
                continue
            if np.linalg.norm(circles[i]-circle[j]) < threshold:
                if votes[i] > votes[j]:
                    circle[j] = 0
                else:
                    circle[i] = 0
    return circle[np.where(circle[:,0]>0)], votes[np.where(circle[:,0]>0)]

if __name__ == "__main__":
    """
    请把目录放在图片目录下
    """

    image_names = ["RTS01org.jpg", "RTS02.jpg", "RTS03.jpg", "RTS04.jpg"]
    # Please change your image here:
    image_name = image_names[1]
    # image_name = "sample003.jpg"
    im = cv2.imread(image_name)
    im[...,[0,2]] = im[...,[2,0]]
    image = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = cv2.GaussianBlur(image, (3,3), 1)

    plt.imshow(im)
    plt.show()
    ##########hyperparameter############
    ransac_iterations = 10000
    votes_threshold = 0.772
    max_r, min_r = 27, 12
    Canny_threshold1, Canny_threshold2 = 100,200
    ####################################
    """
    低于阈值1的像素点会被认为不是边缘；
    高于阈值2的像素点会被认为是边缘；
    在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
    """

    edge = cv2.Canny(im, threshold1 = Canny_threshold1, threshold2 = Canny_threshold2)

    plt.imshow(edge)
    plt.show()

    print("Number of points of interest: {}".format(np.sum(edge>0)))
    """
    True: run detection with RANSAC
    False: run with center oriented detecition method
    """
    random_sample = False

    if random_sample:
        circle_dict = circleOfInterest_random(edge, ransac_iterations, max_r=max_r, min_r=min_r)
        sorted_circle = sorted(circle_dict.items(), key=lambda x: x[1], reverse=True)
        votes = np.array(sorted_circle)[:,1]
        circles = np.array([c[0] for c in sorted_circle])
    else:
        # Fill the (x,y,r) configuration space
        circle_map = circleOfInterest_center_oriented(edge, max_r=max_r, min_r=min_r,
                                                       r_threshold=R_THRESHOLD)
        # Find the 20 best circles, brut force, stupid but easy
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
        votes = np.array(votes)[index_order]

    # Maximum suppression
    new_circles, new_votes = maximum_suppresion(circles, votes, threshold=20)
    print("Top 10 votes: {}".format(votes[:10]))
    print("New votes {}".format(new_votes))
    print(new_votes[0]/new_votes[1])
    res_image = gray2RGB(edge)

    for c in new_circles[np.where(new_votes>max(new_votes)*votes_threshold)]:
        addCircle(res_image, col=c[1], row=c[0], r=c[2])
    fig = plt.figure()
    plt.imshow(res_image)
    plt.show()
    # Convert BGR -  RGB
    res_image[...,[0,2]] = res_image[...,[2,0]]
#    cv2.imwrite("p_{}".format(image_name),res_image)

