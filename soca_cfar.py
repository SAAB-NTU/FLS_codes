import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from scipy.optimize import root
def soca(img, train_hs, guard_hs, tau):
  print(train_hs)
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
            
            "SOCA": soca,
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

img_gray=cv2.imread("test.png",0)

cfar_obj=CFAR(26,10,0.975) 

cfar_result=cfar_obj.detect(img_gray,alg="SOCA")
plt.imshow(cfar_result)

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
        'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(img_gray, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

# Erosion with rectangular kernel
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure
import numpy as np

kernel = np.ones((5, 5), np.uint8)
rec_erosion = cv2.erode(img_gray, kernel, iterations=1)

fd, hog_image = hog(rec_erosion*cfar_result, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(rec_erosion*cfar_result, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# Erosion with eliptical kernel

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
elip_erosion = cv2.erode(img_gray, kernel, iterations=1)

fd, hog_image = hog(elip_erosion*cfar_result, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(elip_erosion*cfar_result, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# Erosion with Cross-Shaped kernel

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
cross_erosion = cv2.erode(img_gray, kernel, iterations=1)

fd, hog_image = hog(cross_erosion*cfar_result, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(cross_erosion*cfar_result, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# Normal Erosion

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure
import numpy as np


kernel = np.array([[0, 1, 0, 0, 0],
                   [1, 0, 1, 0, 0],
                   [0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]], np.uint8)
erosion = cv2.erode(img_gray,kernel,iterations = 1)


fd, hog_image = hog(erosion*cfar_result, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(erosion*cfar_result, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()


# histogram

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure
import numpy as np

fd, hog_image = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img_gray, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
