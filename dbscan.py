import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd

#eps = 5  # Maximum distance between two points to be considered neighbors
#min_samples = 20  # Minimum number of points required to form a dense region
def dbscan_callback(img,eps=6,min_samples=10,img_threshold=50):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(img,img_threshold,255,cv2.THRESH_BINARY)
    coordinates = np.transpose(np.nonzero((th)))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    print("Start")

    dbscan.fit(coordinates)

    # Get the cluster labels assigned to each pixel position
    labels = dbscan.labels_

    # Assign different colors to each cluster
    num_clusters = len(np.unique(labels))  
    # Create a blank colored image
    colored_image = np.zeros_like(img, dtype=np.uint8)
    colored_image=cv2.cvtColor(colored_image,cv2.COLOR_GRAY2BGR)
    # Assign different colors to each cluster
    vars1=[]
    vars2=[]
    centres=[]
    print("Ended!")
    for label in range(num_clusters):

        cluster_pixels = coordinates[labels == label]

        if(len(cluster_pixels)>int(1)):


            process=np.asarray(cluster_pixels)

            vars1.append(np.var(process[:,0]))
            vars2.append(np.var(process[:,1]))

            centres.append((np.array(np.mean(process[:,1]),np.uint),np.array(np.mean(process[:,0]),np.uint)))

            color = np.array(np.random.randint(0, 256, size=3).tolist())  # Generate a random color for each cluster

            for pixel in cluster_pixels:

                colored_image[pixel[0]][pixel[1]] = color
                
    #print(centres[vars1.index(min(vars1))][0])
    #print(centres[vars2.index(min(vars2))][1])
    #print(min(vars1))
    #print(min(vars2))
    #print(num_clusters)
    centre=(int(centres[vars1.index(min(vars1))][0]),int(centres[vars1.index(min(vars1))][1]))
    #print(vars1)
    #print(vars2)
    #print((centres[vars2.index(min(vars2))]))
    #centre=(100,100)
    axes_lengths=(int(min(vars2)/2),int(min(vars1)/2))
    #print(axes_lengths)
    color_e = (0, 255, 0)  # Green color in BGR
    thickness = 2
    #print(axes_lengths)
    cv2.circle(colored_image, centre, radius=5, color=(255, 100, 0), thickness=1)
    cv2.ellipse(colored_image, centre, axes_lengths, angle=0, startAngle=0, endAngle=360, color=color_e, thickness=thickness)
    
    # return colored_image,centre,axes_lengths


    cv2.imshow('DBScan', colored_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()