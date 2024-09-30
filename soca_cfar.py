import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import time

from scipy.optimize import root
from skimage.measure import block_reduce

def soca(img, train_hs, guard_hs, tau):
  """
  Implements Sum-Oriented Constant Amplitude filter based on NumPy.

  Args:
    img: Input image as a NumPy array (assumed to be BGR or grayscale).
    train_hs: Training window half-size.
    guard_hs: Guard window half-size.
    tau: Threshold parameter.

  Returns:
    A 2D NumPy array with filtered binary image.
  """

  # Convert OpenCV image to NumPy array and grayscale (if necessary)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  blurred_image = cv2.GaussianBlur(img_gray, (7, 7), 0)
  denoised_image = cv2.fastNlMeansDenoising(blurred_image, h=200)


  rows, cols = denoised_image.shape
  ret = np.zeros((rows, cols), dtype=np.uint8)

  for col in range(cols):
    for row in range(train_hs + guard_hs, rows - train_hs - guard_hs):
      leading_sum, lagging_sum = 0.0, 0.0
      for i in range(row - train_hs - guard_hs, row + train_hs + guard_hs + 1):
        if (i - row) > guard_hs:
          lagging_sum += denoised_image[i, col]
        else:
          leading_sum += denoised_image[i, col]
      sum_train = np.min([leading_sum, lagging_sum])
      ret[row, col] = (tau * sum_train / train_hs)

  return ret

def soca_new(img, train_hs, guard_hs, tau):
  start_time = time.time()
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  blurred_image = cv2.GaussianBlur(img_gray, (5, 5), 0)
  denoised_image = cv2.fastNlMeansDenoising(img_gray, h=200)
  
  max_pooled_image = block_reduce(denoised_image, (2,2),np.max)

  integral_image = cv2.integral(max_pooled_image)

  # Remove 1st row and 1st column
  trimmed_image = integral_image[1:,1:]

  rows, cols = trimmed_image.shape
  ret = np.zeros((rows, cols), dtype=np.uint8)

  for col in range(train_hs + guard_hs, cols - train_hs - guard_hs):
    for row in range(train_hs + guard_hs, rows - train_hs - guard_hs):
      
      leading_guard = calc_rect_sum(trimmed_image,
                                    row - guard_hs, col - guard_hs,
                                    guard_hs, 2*guard_hs + 1)
      leading_sum = calc_rect_sum(trimmed_image,
                                  row - guard_hs - train_hs, col - guard_hs - train_hs,
                                  guard_hs + train_hs, 2*guard_hs + 2*train_hs + 1)
      leading_train = leading_sum - leading_guard
      
      lagging_guard = calc_rect_sum(trimmed_image,
                                    row - guard_hs, col + 1,
                                    guard_hs, 2*guard_hs + 1)
      lagging_sum = calc_rect_sum(trimmed_image,
                                  row - guard_hs - train_hs, col + 1,
                                  guard_hs + train_hs, 2*guard_hs + 2*train_hs + 1)
      lagging_train = lagging_sum - lagging_guard 
      sum_train = np.min([leading_train, lagging_train])
      total_train_cells = train_hs*(2*train_hs + 2*guard_hs + 1)
      ret[row, col] = (tau * sum_train / total_train_cells)
  print(f"soca_new duration: {time.time() - start_time}")

  return ret

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

def goca(img, train_hs, guard_hs, tau):
  """
  Implements Sum-Oriented Constant Amplitude filter based on NumPy.

  Args:
    img: Input image as a NumPy array (assumed to be BGR or grayscale).
    train_hs: Training window half-size.
    guard_hs: Guard window half-size.
    tau: Threshold parameter.

  Returns:
    A 2D NumPy array with filtered binary image.
  """

  # Convert OpenCV image to NumPy array and grayscale (if necessary)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  blurred_image = cv2.GaussianBlur(img_gray, (7, 7), 0)
  denoised_image = cv2.fastNlMeansDenoising(blurred_image, h=200)

  rows, cols = denoised_image.shape
  ret = np.zeros((rows, cols), dtype=np.uint8)

  for col in range(cols):
    for row in range(train_hs + guard_hs, rows - train_hs - guard_hs):
      leading_sum, lagging_sum = 0.0, 0.0
      for i in range(row - train_hs - guard_hs, row + train_hs + guard_hs + 1):
        if (i - row) > guard_hs:
          lagging_sum += denoised_image[i, col]
        else:
          leading_sum += denoised_image[i, col]
      sum_train = np.max([leading_sum, lagging_sum])
      ret[row, col] = ((tau * sum_train / train_hs))

  return ret
class CFAR(object):
      """
      Constant False Alarm Rate (CFAR) detection with several variants
          - Cell averaging (CA) CFAR
          - Greatest-of cell-averaging (GOCA) CFAR
          - Order statistic (OS) CFAR
      """

      def __init__(self, Ntc, Ngc, Pfa, rank=None):
          self.Ntc = Ntc #number of training cells
          assert self.Ntc % 2 == 0
          self.Ngc = Ngc #number of guard cells
          assert self.Ngc % 2 == 0
          self.Pfa = Pfa #false alarm rate
          if rank is None: #matrix rank
              self.rank = self.Ntc / 2
          else:
              self.rank = rank
              assert 0 <= self.rank < self.Ntc

          #threshold factor calculation for the 4 variants of CFAR
          self.threshold_factor_SOCA = self.calc_WGN_threshold_factor_SOCA()
          self.threshold_factor_GOCA = self.calc_WGN_threshold_factor_GOCA()


          self.params = {
              
              "SOCA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_SOCA),
              "GOCA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_GOCA),

          }
          self.detector = {
              
              "SOCA": soca_new,
              "GOCA": goca,
          
          }
        

      def __str__(self):
          return "".join(
              [
                  "CFAR Detector Information\n",
                  "=========================\n",
                  "Number of training cells: {}\n".format(self.Ntc),
                  "Number of guard cells: {}\n".format(self.Ngc),
                  "Probability of false alarm: {}\n".format(self.Pfa),
                  "Order statictics rank: {}\n".format(self.rank),
                  "Threshold factors:\n",
                
                  "    SOCA-CFAR: {:.3f}\n".format(self.threshold_factor_SOCA),
                  "    GOCA-CFAR: {:.3f}\n".format(self.threshold_factor_GOCA),
          
              ]
          )

      def calc_WGN_threshold_factor_CA(self):
          return self.Ntc * (self.Pfa ** (-1.0 / self.Ntc) - 1)

      def calc_WGN_threshold_factor_SOCA(self):
          x0 = self.calc_WGN_threshold_factor_CA()
          for ratio in np.logspace(-2, 2, 10):
              ret = root(self.calc_WGN_pfa_SOCA, x0 * ratio)
              if ret.success:
                  return ret.x[0]
          raise ValueError("Threshold factor of SOCA not found")

      def calc_WGN_threshold_factor_GOCA(self):
          x0 = self.calc_WGN_threshold_factor_CA()
          for ratio in np.logspace(-2, 2, 10):
              ret = root(self.calc_WGN_pfa_GOCA, x0 * ratio)
              if ret.success:
                  return ret.x[0]
          raise ValueError("Threshold factor of GOCA not found")

      def calc_WGN_pfa_GOSOCA_core(self, x):
          x = float(x)
          temp = 0.0
          for k in range(int(self.Ntc / 2)):
              l1 = math.lgamma(self.Ntc / 2 + k)
              l2 = math.lgamma(k + 1)
              l3 = math.lgamma(self.Ntc / 2)
              temp += math.exp(l1 - l2 - l3) * (2 + x / (self.Ntc / 2)) ** (-k)
          return temp * (2 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)

      def calc_WGN_pfa_SOCA(self, x):
          return self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

      def calc_WGN_pfa_GOCA(self, x):
          x = float(x)
          temp = (1.0 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)
          return temp - self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2


      def detect(self, mat, alg="CA"):
          """
          Return target mask array.
          """
          #pad_width = [(self.Ntc//2 + self.Ngc//2, self.Ntc//2 + self.Ngc//2), (self.Ntc//2 + self.Ngc//2, self.Ntc//2 + self.Ngc//2)]
          #mat=np.pad(mat, pad_width=pad_width, mode='constant', constant_values=0)
          return self.detector[alg](mat, *self.params[alg])

def soca_code(soca_trial, ntc, ngc, pfa):

  # img_gray=cv2.imread("test.png",0)
  img = soca_trial
  cfar_obj = CFAR(ntc,ngc,pfa) 

  cfar_result = cfar_obj.detect(img,alg="SOCA")

  del cfar_obj

  return cfar_result*255


if __name__ == "__main__":
  path = "/home/trgknng/2023_UWR/Analysis/raw_data/26Jan/FLS/fls_images_polar"
  image_name = "1706238550866219979.png"
  image_path = os.path.join(path,image_name)
  
  image = cv2.imread(image_path)

  if image is None:
     print("Error importing image")
     exit()

  # Optimal parameters up to this point: 12, 4, 0.97
  # ntc_list = [4,6,8,10,12,14,16,18,20] 
  # ngc_list = [4,6,8,10,12,14,16,18,20] 
  ngc = 4
  ntc = 12
  pfa = 0.97
  result_filename = f"cfar_results/{int(ntc)}_{int(ngc)}_{int(pfa*1000)}.png"
  result_path = os.path.join(os.getcwd(),result_filename)
  print(result_path)
  result = soca_code(image, ntc, ngc, pfa)
  # cv2.imwrite(result_path,result)
  # cv2.imshow("original image", image)
  cv2.imshow('SOCA_CFAR', result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()