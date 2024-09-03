import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd

from convex_hull import convex_hull

# Load an image
trial_test = cv2.imread('/home/t1/Desktop/0001image.png')

# convex_hull()

cv2.imshow('Trial_Image', trial_test)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Read an image using OpenCV
# trial = cv2.imread('/home/t1/Desktop/0001image.png')

# Convert the image from BGR to RGB (OpenCV loads images in BGR format)
# img_rgb = cv2.cvtColor(trial, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
# plt.imshow(img_rgb)
# plt.axis('off')  # Hide axis
# plt.show()

# # Open an image using PIL
# img = Image.open('/home/t1/Documents/Sonar Repositories/Visual Studio Code/AUV')

# # Display the image
# plt.imshow(img)
# plt.axis('off')  # Hide axis
# plt.show()

