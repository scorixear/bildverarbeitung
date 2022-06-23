import sys, os
from tabnanny import check
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def kernel(img, matrix):
  newimg = np.zeros(img.shape)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      fits = True
      for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
          if matrix[y,x] != 2:
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
  return newimg

def dilute(img):
  kernelImages = []
  print("Kernel 1")
  kernelImages.append(kernel(img, np.array([[0,2,1],[0,1,1],[0,2,1]])))
  print("Kernel 2")
  kernelImages.append(kernel(img, np.array([[0,0,2],[0,1,1],[2,1,1]])))
  print("Kernel 3")
  kernelImages.append(kernel(img, np.array([[0,0,0],[2,1,2],[1,1,1]])))
  print("Kernel 4")
  kernelImages.append(kernel(img, np.array([[2,0,0],[1,1,0],[1,1,2]])))
  print("Kernel 5")
  kernelImages.append(kernel(img, np.array([[1,2,0],[1,1,0],[1,2,0]])))
  print("Kernel 6")
  kernelImages.append(kernel(img, np.array([[1,1,2],[1,1,0],[2,0,0]])))
  print("Kernel 7")
  kernelImages.append(kernel(img, np.array([[1,1,1],[2,1,2],[0,0,0]])))
  print("Kernel 8")
  kernelImages.append(kernel(img, np.array([[2,1,1],[0,1,1],[0,0,2]])))
  newimg = np.zeros(img.shape)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      Value = 0
      for k in kernelImages:
        if k[i,j] == 1:
          Value = 1
          break
      newimg[i,j]= Value
  return newimg
    
  

def checkDifference(img1, img2):
  sum = 0
  for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
      if img1[i,j] != img2[i,j]:
        sum += 1
  return sum

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'cat2.jpg'))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if img.max() <= 1:
    img = img * 255
  img = img.astype('uint8')

  oldPic = np.where(img > 127, 1, 0).astype('uint8')
  binary = np.where(img > 127, 1, 0).astype('uint8')
  
  diluted = dilute(oldPic)
  diff = checkDifference(oldPic, diluted)
  while diff > 200:
    plt.subplot(121)
    plt.imshow(oldPic, cmap='gray')
    plt.subplot(122)
    plt.imshow(diluted, cmap='gray')
    plt.show()
    print("Recalculating dilute",diff)
    oldPic = diluted
    diluted = dilute(oldPic)
    diff = checkDifference(oldPic, diluted)
    
    
  plt.subplot(121)
  plt.imshow(binary, cmap="gray")
  plt.title("Original")
  plt.subplot(122)
  plt.imshow(diluted, cmap="gray")
  plt.title("Diluted")
  plt.show()
  

if __name__ == "__main__":
  main()