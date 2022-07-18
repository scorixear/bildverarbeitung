import sys, os
from tabnanny import check
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv

def kernel(img, matrix):
  newimg = np.zeros(img.shape)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      fits = True
      for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
          if matrix[y,x] != None:
            if i+y-1 < 0 or i+y-1 >= img.shape[0] or j+x-1 < 0 or j+x-1 >= img.shape[1]:
              if matrix[y,x] != 0:
                fits = False
                break
            elif matrix[y,x] != img[i+y-1,j+x-1]:
              fits = False
              break
        if not fits:
          break
      if fits:
        newimg[i,j] = 1
  return newimg.astype('uint8')
def cv_kernel(img, matrix):
  matrix[matrix == 0] = -1
  matrix[matrix == None] = 0
  return cv.morphologyEx(img, cv.MORPH_HITMISS, matrix.astype(int))

def thin(img):
  newimg = cv.bitwise_and(img,    cv.bitwise_not(cv_kernel(img, np.array([[0,None,1],[0,1,1],[0,None,1]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[0,0,None],[0,1,1],[None,1,1]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[0,0,0],[None,1,None],[1,1,1]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[None,0,0],[1,1,0],[1,1,None]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,None,0],[1,1,0],[1,None,0]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,1,None],[1,1,0],[None,0,0]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,1,1],[None,1,None],[0,0,0]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[None,1,1],[0,1,1],[0,0,None]]))))
  return newimg
    


def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'cat.jpg'))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if img.max() <= 1:
    img = img * 255
  img = img.astype('uint8')

  binary = np.where(img > 127, 0, 1).astype('uint8')
  mpimg.imsave(os.path.join(os.path.dirname(__file__), 'binary1.png'), binary, cmap='gray')
  #binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3,3), dtype='uint8'), iterations=3)
  #binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((3,3), dtype='uint8'), iterations=3)
  #binary = cv.bitwise_not(binary)
  #binary = cv.erode(binary, np.ones((3,3), np.uint8), iterations=3)
  oldPic = binary.copy()
  
  
  thinned = thin(oldPic)
  zeros = cv.countNonZero(thinned)
  prevZeros = cv.countNonZero(oldPic)
  while zeros != 0 and zeros != prevZeros:
    print("Recalculating thinning",zeros)
    oldPic = thinned
    thinned = thin(oldPic)
    prevZeros = zeros
    zeros = cv.countNonZero(thinned)
    
  mpimg.imsave(os.path.join(os.path.dirname(__file__), 'thinned1.png'), oldPic, cmap='gray')
  plt.subplot(121)
  plt.imshow(binary, cmap="gray")
  plt.title("Original")
  plt.subplot(122)
  plt.imshow(oldPic, cmap="gray")
  plt.title("Thinned")
  plt.show()
  

if __name__ == "__main__":
  main()