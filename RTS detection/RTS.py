# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:04:10 2019

@author: TeTe
"""
#%%
import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from collections import defaultdict
from tqdm import tqdm
#%%
def threePointsToCircle(points_x, points_y):
    """
    https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
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
#%%
def numPointsOnCircle(edge, col, row, r, threshold_r):
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
                dis = np.sqrt((j-row)**2 + (i-col)**2)
                if dis>=inner_r and dis<=outer_r:
                    ans += 1
    return ans

def circleOfInterest(edge, n_sample, max_r):
    circle_dict = defaultdict(int)
    x, y = np.where(edge>0)
    num_points = len(x)
    print("number of points of interest {}".format(num_points))
    for i in tqdm(range(n_sample)):
        indexes = random.sample(range(0, num_points), k=3)
        points_x, points_y = x[indexes], y[indexes]
        h, k, r = threePointsToCircle(points_x, points_y)
        # h : row, k: col
        if r > max_r:
            continue
        r_threshold = 3
        num_points_on_circle = numPointsOnCircle(edge, row=h, col=k , r=r, 
                                                 threshold_r=r_threshold)
        if num_points_on_circle > 0:
            circle_dict[(int(h), int(k), int(r))] += num_points_on_circle
    
    return circle_dict

def gray2RGB(gray):
    h, w = gray.shape
    rgb = np.zeros((h, w, 3))
    new_gray = (0.8 * gray).astype(int)
    rgb[..., 0] = new_gray
    rgb[..., 1] = new_gray
    rgb[..., 2] = new_gray
    rgb = rgb.astype(int)
    return rgb

def addCircle(im, col, row, r, delta_r=3, c=(255, 0, 0)):
    inner_r = r - delta_r
    outer_r = r + delta_r
    h, w = edge.shape
    # h: row, w: col
    # x: col, y: row
    for j in range(max(0, int(col-outer_r)), min(w, int(col+outer_r))):  # col
        for i in range(max(0, int(row-outer_r)), min(h, int(row+outer_r))): # row
            dis = np.sqrt((j-col)**2 + (row-i)**2)
            if dis>=inner_r and dis<=outer_r:
                im[i, j] = c
    return 
#%%


if __name__ == "__main__":
    image_name = "sample005.jpg"
    im = cv2.imread(image_name)
    im[...,[0,2]] = im[...,[2,0]]
    plt.imshow(im) 
#%%
    """
    低于阈值1的像素点会被认为不是边缘；
    高于阈值2的像素点会被认为是边缘；
    在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
    """
    edge = cv2.Canny(im, threshold1 = 300, threshold2 = 450)
    plt.imshow(edge)
#%%
    print("Number of points of interest: {}".format(np.sum(edge)))
    circle_dict = circleOfInterest(edge, 10000, max_r=50)
    # key: row, col, radius
#%%
    sorted_circle = sorted(circle_dict.items(), key=lambda x: x[1], reverse=True)
#%%
    res_image = gray2RGB(edge)
    for circle in sorted_circle:
        c = circle[0]
        addCircle(res_image, col=c[1], row=c[0], r=c[2])
    plt.imshow(res_image)
#%%
    plt.imshow(edge)
#%%
#t1 = time.time()
#for i in range(1000000):
#    k = random.sample(range(0, 30000), k=3)
#t2 = time.time()
#print(t2-t1)
#    
##%%
#t1 = time.time()
#for i in range(10000):
#    k = random.shuffle(list(range(0, 30000)))
#t2 = time.time()
#print(t2-t1)
##%%
#n = 30000
#t1 = time.time()
#for i in range(1000000):
#    a = random.randint(0, n-1)
#    b = random.randint(0, n-1)
#    while a == b:
#        b = random.randint(0, n-1)
#    c = random.randint(0, n-1)
#    while b == c or a == c:
#        c = random.randint(0, n-1)     
#t2 = time.time()
#print(t2-t1)
