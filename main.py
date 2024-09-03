import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd

# Import Functions
from dbscan import dbscan_callback
from contour_drawing import contour_drawing
from convex_hull import convex_hull
from distance_with_points import dist_with_points
from shortest_distance import shortest_distance
from video_writer import video_writer

# System Arguments


def main():
    trial = cv2.imread('/home/t1/Desktop/0001image.png')
    dbscan_callback(img=trial)
    convex_hull(trial)
    shortest_distance(trial)
    dist_with_points(trial)

main()