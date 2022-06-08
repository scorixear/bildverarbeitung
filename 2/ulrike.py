import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.special

def getGauss(m: int):
  matrix = np.zeros((m+1, m+1))
  for i in range(m+1):
    for j in range(m+1):
      matrix[i,j] = scipy.special.binom(m, i)*scipy.special.binom(m, j)
  return matrix * (1/(2**(2*m)))

def main():
  image = plt.imread('C:/Users/pk/Dropbox/Master/MMI/2. Semester/Bildverarbeitung/Aufgaben/2/test2.jpg')
  gauss = getGauss(4)
  filtered = cv.filter2D(image, -1, gauss)
  plt.imshow(filtered)
  plt.show()

if __name__ == "__main__":
  sys.exit(main())
