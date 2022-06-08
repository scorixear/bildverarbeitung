import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def a(img: np.ndarray, cv2img):
  hist, bins = np.histogram(img.flatten(),256,[0,256])
  equ = cv2.equalizeHist(cv2img)
  plt.subplot(221)
  plt.imshow(img, cmap='gray')
  plt.subplot(222)
  plt.imshow(equ, cmap='gray')
  plt.subplot(223)
  plt.hist(img.flatten(), 256, [0,256], color='r')

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__),"moonlanding.png"))
  img = img*255
  cv2img = cv2.imread(os.path.join(os.path.dirname(__file__),"moonlanding.png"))
  cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
  a(img, cv2img)
  plt.show()

if __name__ == "__main__":
  sys.exit(main())