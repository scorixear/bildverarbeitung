import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'cat.jpg'))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if img.max() <= 1:
    img = img * 255
  img = img.astype('uint8')

  binary = np.where(img > 127, 1, 0).astype('uint8')
  eroded = cv.erode(binary, np.ones((3,3)).astype('uint8'))
  dilated = cv.dilate(binary, np.ones((3,3), np.uint8))
  opening = cv.dilate(eroded, np.ones((3,3), np.uint8))
  #opening_real = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
  closing = cv.erode(dilated, np.ones((3,3), np.uint8))
  #closing_real = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))

  plt.subplot(2, 3, 1)
  plt.imshow(img, cmap="gray")
  plt.title("Original")
  plt.subplot(2, 3, 2)
  plt.imshow(binary, cmap="gray")
  plt.title("Binary")
  plt.subplot(2, 3, 3)
  plt.imshow(eroded, cmap="gray")
  plt.title("Eroded")
  plt.subplot(2, 3, 4)
  plt.imshow(dilated, cmap="gray")
  plt.title("Dilated")
  plt.subplot(2, 3, 5)
  plt.imshow(opening, cmap="gray")
  plt.title("Opening")
  plt.subplot(2, 3, 6)
  plt.imshow(closing, cmap="gray")
  plt.title("Closing")
  plt.show()


if __name__ == "__main__":
  main()