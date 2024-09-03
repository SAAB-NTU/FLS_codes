
#convex hull

import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd
import random as rng
 
def convex_hull(trial):
    rng.seed(12345)

    # Load an image
    # trial = cv2.imread('/home/t1/Desktop/0001image.png')

    # Read image
    convexhull_image = trial
    gray_convexhull=cv2.cvtColor(convexhull_image, cv2.COLOR_BGR2GRAY)

    threshold = 100

    # Detect edges using Canny
    canny_output = cv2.Canny(gray_convexhull, threshold, threshold * 2)

    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plt.figure(figsize=(14,14))

    # Find the convex hull object for each contour
    hull_list = []
    moments_list=[]
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
        moments_list.append(cv2.moments(contours[i]))
        

    # Draw contours + hull results
    convex_hull_output = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(convex_hull_output, contours, i, color)
        cv2.drawContours(convex_hull_output, hull_list, i, color)
    #     try:
    #         cX = int(moments_list[i]["m10"] / moments_list[i]["m00"])
    #         cY = int(moments_list[i]["m01"] / moments_list[i]["m00"])
    #         a=0
    #         b=0
    #         if(cX<300):
    #             a=cX+50
    #             b=cY-50
    #         else:
    #             a=cX-50
    #             b=cY-50
    #         cv2.circle(drawing_with_dist_pts,(a, b), 7, (255, 0, 255), -1)
    #     except:
    #         print("division by zero")

    # Display the image
    # plt.imshow(drawing_with_dist_pts)
    # Display the image in a new window
    cv2.imshow('Convex Hull Image', convex_hull_output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# convex_hull()
