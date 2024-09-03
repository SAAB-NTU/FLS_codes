import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd
import random as rng

def contour_drawing():
    trial="/home/t1/Documents/Sonar Repositories/FLS/Soca Processed Images/0001image.png"
    # img = cv2.imread(trial)
    # plt.imshow(img)



    # Load the image
    near_surf_img = cv2.imread(trial, 0)
    #near_surf_img = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary_image = cv2.threshold(near_surf_img, 127, 255, cv2.THRESH_BINARY)

    # Invert the binary image if necessary
    #binary_image = cv2.bitwise_not(binary_image)

    # Find contours in the binary image
    ns_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the point (x, y) from which you want to find the nearest surface
    point = (200, 350)  # Replace with your coordinates

    # Initialize the minimum distance and the nearest contour point
    min_distance = float('inf')
    nearest_point = None

    # Iterate through each contour and find the nearest point
    for ns_contour in ns_contours:
        for contour_point in ns_contour:
            contour_point = contour_point[0]
            distance = np.linalg.norm(np.array(point) - np.array(contour_point))
            if distance < min_distance:
                min_distance = distance
                nearest_point = tuple(contour_point)

    # Draw contours, the given point, and the nearest point on the original image for visualization
    ns_output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(ns_output_image, ns_contours, -1, (0, 255, 0), 2)
    cv2.circle(ns_output_image, point, 5, (0, 0, 255), -1)
    if nearest_point:
        cv2.circle(ns_output_image, nearest_point, 5, (255, 0, 0), -1)
        cv2.line(ns_output_image, point, nearest_point, (255, 0, 0), 1)

    plt.figure(figsize=(14,14))    
        
        
    # Display the image
    cv2.imshow("Hi",ns_output_image)

    # print(f"The nearest point on the surface from {point} is: {nearest_point} with a distance of {min_distance}")
