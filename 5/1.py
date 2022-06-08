import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'angrycat.jpg'))
  ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
  plt.subplot(121)
  plt.imshow(img)
  plt.subplot(122)
  plt.imshow(thresh)

  plt.show()


if __name__ == "__main__":
  main()