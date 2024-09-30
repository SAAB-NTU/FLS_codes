import cv2
import os
import time
# from skimage.measure import block_reduce
import skimage
import skimage.measure
import numpy as np
from matplotlib import pyplot as plt


path = "/home/trgknng/2023_UWR/Analysis/raw_data/26Jan/FLS/fls_images_polar"

def calc_rect_sum(integral, x,y,w,h):
  """
   Calculate the block sum of an integral image

   Args:
    integral: the integral image to calculate rectangular sum
    x: upperleft x_coordinate, heading downwards for images
    y: upperleft y_coordinate, heading rightwards for images
    w: width
    h: height

   Returns:
    Sum of the rectangular block in integral image
  """
  return integral[x+h-1,y+w-1] - integral[x-1,y+w-1] - integral[x+h-1,y-1] + integral[x-1,y-1]

if __name__ == "__main__":
  image_name = "1706238550866219979.png"
  image_path = os.path.join(path,image_name)
  image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

  Ntc = 20
  Ngc = 3
  row = 252
  col = 317

  rows, cols = image.shape
  if image is None:
     print("Error importing image")
     exit()
  # blurred_image = cv2.GaussianBlur(image,(15,15),0)
  # denoised_image = cv2.fastNlMeansDenoising(image, h=100)
  start_time = time.time()
  min_pool = skimage.measure.block_reduce(image,(4,4),np.max)
  # print(f"duration: {time.time() - start_time}")
  cv2.imshow("min_pool", min_pool)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # plt.imshow(min_pool)
  # plt.show()
  # print(min_pool)

  # Display original and reduced images
  # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

  # # Original image
  # ax[0].imshow(image, cmap='gray')
  # ax[0].set_title('Original Image')
  # ax[0].axis('off')  # Hide axes

  # # Reduced image
  # ax[1].imshow(min_pool, cmap='gray')
  # ax[1].set_title('Reduced Image')
  # ax[1].axis('off')  # Hide axes

  # plt.show()