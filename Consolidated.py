import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd


#Cv2 bridge ros

#No need of ROI, video


#Graphs --> Nodes and edges

# Function to select ROI in the first frame
def select_roi(frame):
    roi = cv2.selectROI(frame)
    cv2.destroyAllWindows()  # Close the window after ROI selection
    return roi


# Function to crop frames using the selected ROI
def crop_frames_video(video_path, output_folder, roi):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break  # Break the loop if we reach the end of the video

        # Crop frame using the selected ROI
        x, y, w, h = roi
        cropped_frame = frame[y:y + h, x:x + w]

        # Save the cropped frame to the output folder
        output_path = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(output_path, cropped_frame)

        frame_count += 1

    video_capture.release()

def crop_frames(img_paths, output_folder, roi):
    for img_path in img_paths:
        # Crop frame using the selected ROI
        frame=cv2.imread(img_path,0)
        x, y, w, h = roi
        cropped_frame = frame[y:y + h, x:x + w]

        # Save the cropped frame to the output folder
        output_path = os.path.join(output_folder,img_path.split("/")[-1])
        cv2.imwrite(output_path, cropped_frame)

        
video_path="/home/t1/Documents/Sonar Repositories/FLS/Soca Processed Images/"
output_folder="/home/t1/Documents/Sonar Repositories/FLS/DB Scan Output/"
#video_path = 'C:\\Users\\abuba\\OneDrive - Nanyang Technological University\\PhD_Work\\Thesis_sem_6\\FYP Ureca Sem 2\\Experiments\\19 Jan\\FLS\\20240119_111638_exported.avi'  # Replace with your video file
#output_folder = 'C:\\Users\\abuba\\OneDrive - Nanyang Technological University\\PhD_Work\\Thesis_sem_6\\FYP Ureca Sem 2\\Experiments\\19 Jan\\FLS\\output_frames_20240119_111638_exported'
img_list=sorted(glob(video_path+"/*"))
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


first_frame=cv2.imread(img_list[15])
#video_capture = cv2.VideoCapture(video_path)
#ret, first_frame = video_capture.read()
#video_capture.release()

# Select ROI in the first frame
roi = select_roi(first_frame)

crop_frames(img_list, output_folder, roi)

#eps = 5  # Maximum distance between two points to be considered neighbors
#min_samples = 20  # Minimum number of points required to form a dense region
def dbscan_callback(img,eps=6,min_samples=10,img_threshold=50):
    _,th=cv2.threshold(img,img_threshold,255,cv2.THRESH_BINARY)
    coordinates = np.transpose(np.nonzero((th)))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)



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
    
    return colored_image,centre,axes_lengths

from glob import glob #helps with finding all directories at once
paths=sorted(glob("/home/t1/Documents/Sonar Repositories/FLS/DB Scan Output/*.png"))

counter=0
centres=[]
axes_lengths=[]
for path in paths:
    sample_img=cv2.imread(path,0)
    from scipy import ndimage
    sample_img = ndimage.rotate(sample_img, 0)
    #sample_img=cv2.rotate(sample_img,30)
    r,t=cv2.threshold(sample_img,20,255,cv2.THRESH_BINARY)
    res,x,y=dbscan_callback(sample_img)
    centres.append(x)
    axes_lengths.append(y)
    #plt.imshow(res)
    sample_img=cv2.cvtColor(sample_img,cv2.COLOR_GRAY2BGR)
    stack=np.hstack((sample_img,res))
    cv2.imwrite("/home/t1/Documents/Sonar Repositories/FLS/DB Scan Output/"+f"{counter:04d}"+"image.png",stack)
    counter=counter+1

import os
import imageio

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Set video dimensions (width and height)
width, height = stack.shape[1], stack.shape[0]

video = cv2.VideoWriter('/home/t1/Documents/Sonar Repositories/FLS/Output Video/output_video.mp4', fourcc, 20, (width, height))

# Path to the directory containing your photos
photos_dir = '/home/t1/Documents/Sonar Repositories/FLS/DB Scan Output/'

# Function to sort files by name
def sort_files_by_name(files):
    return sorted(files, key=lambda x: (os.path.splitext(x)[0]))

# Get the list of photo files in the directory
photo_files = [f for f in os.listdir(photos_dir) if os.path.isfile(os.path.join(photos_dir, f))]
photo_files = sort_files_by_name(photo_files)

for photo_file in photo_files[:]:
    photo_path = os.path.join(photos_dir, photo_file)
    frame = cv2.imread(photo_path)
    video.write(frame)

video.release()
print('Video created successfully!')

trial="/home/t1/Documents/Sonar Repositories/FLS/Soca Processed Images/0001image.png"
img = cv2.imread(trial)
plt.imshow(img)



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

print(f"The nearest point on the surface from {point} is: {nearest_point} with a distance of {min_distance}")

#convex hull

import random as rng
 
rng.seed(12345)

# Read image
convexhull_image = cv2.imread(trial)
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
drawing_with_dist_pts = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing_with_dist_pts, contours, i, color)
    cv2.drawContours(drawing_with_dist_pts, hull_list, i, color)
    try:
        cX = int(moments_list[i]["m10"] / moments_list[i]["m00"])
        cY = int(moments_list[i]["m01"] / moments_list[i]["m00"])
        a=0
        b=0
        if(cX<300):
            a=cX+50
            b=cY-50
        else:
            a=cX-50
            b=cY-50
        cv2.circle(drawing_with_dist_pts,(a, b), 7, (255, 0, 255), -1)
    except:
        print("division by zero")

# Display the image
plt.imshow(drawing_with_dist_pts)

#shortest distance

# Draw contours + hull results
def shortest_distance(trial):
    
    # Read image
    convexhull_image = cv2.imread(trial)
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


    drawing_with_dist_pts = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    centres = []
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing_with_dist_pts, contours, i, color)
        cv2.drawContours(drawing_with_dist_pts, hull_list, i, color)
        try:
            cX = int(moments_list[i]["m10"] / moments_list[i]["m00"])
            cY = int(moments_list[i]["m01"] / moments_list[i]["m00"])
            a=0
            b=0
            if(cX<300):
                a=cX+50
                b=cY-50
            else:
                a=cX-50
                b=cY-50
            cv2.circle(drawing_with_dist_pts,(a, b), 7, (255, 0, 255), -1)
            centres.append((a,b))
        except:
            print("division by zero")
    return centres




#Distance Plus Points

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
points = shortest_distance(trial) # Replace with your coordinates

# Initialize the minimum distance and the nearest contour point


# Iterate through each contour and find the nearest point
ns_output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(ns_output_image, ns_contours, -1, (0, 255, 0), 2)
for point in points[:]:
    min_distance = float('inf')
    nearest_point = None
    for ns_contour in ns_contours:
        for contour_point in range (len(ns_contour)):
            ns_contour[contour_point] = ns_contour[contour_point][0]
            distance = np.linalg.norm(np.array(point) - np.array(ns_contour[contour_point]))
            if distance < min_distance:
                min_distance = distance
                nearest_point = tuple(ns_contour[contour_point])
                nearest_point_index = contour_point
        cv2.circle(ns_output_image, point, 5, (0, 0, 255), -1)
    if nearest_point:
        cv2.circle(ns_output_image, nearest_point[0], 5, (255, 0, 0), -1)
        cv2.line(ns_output_image, point, nearest_point[0], (255, 0, 0), 1)
        for index in range (nearest_point_index -10, nearest_point_index +10):
            cv2.circle(ns_output_image, ns_contour[index][0], 5, (255, 0, 0), -1)
            


plt.figure(figsize=(14,14))    
    
    
# Display the image
plt.imshow(ns_output_image)

print(f"The nearest point on the surface from {point} is: {nearest_point} with a distance of {min_distance}")





# from cv_bridge import CvBridge
# bridge = CvBridge()
# image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")