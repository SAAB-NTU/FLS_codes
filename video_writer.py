
def video_writer():
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