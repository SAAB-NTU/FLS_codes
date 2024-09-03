import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from glob import glob
import pandas as pd

def roi():
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