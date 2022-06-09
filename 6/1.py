import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def laplace(img):
  return kernel_transform(img, np.array([[0,1,0],[1,-4,1],[0,1,0]]))
def kernel_transform(img, kernel):
  return cv.filter2D(img, -1, kernel)

def main():
  plt_rows = 1
  plt_columns = 2
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'catshop.jpg'))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  images = [img]
  for i in range(3):
    images.append(cv.resize(images[i], [images[i].shape[1]//2, images[i].shape[0]//2], interpolation=cv.INTER_NEAREST))
  laplaceImages = []
  for i in range(len(images)):
    laplaceImages.append(laplace(images[i]))
  reconstructedImages = []
  for i in range(len(images)-1, 0, -1):
    resize = cv.resize(images[i], [laplaceImages[i-1].shape[1], laplaceImages[i-1].shape[0]], interpolation=cv.INTER_NEAREST)
    laplaceImage = laplaceImages[i-1]
    reconstructed = (resize + laplaceImage)/2
    reconstructedImages.append((resize, laplaceImage, reconstructed))
  imageCounter = 1
  for i in range(len(images)):
    plt.subplot(plt_rows, plt_columns,imageCounter)
    plt.title('Gaus '+str(i))
    plt.imshow(images[i], cmap="gray")
    imageCounter+=1
    plt.subplot(plt_rows, plt_columns,imageCounter)
    plt.title('Laplace '+str(i))
    plt.imshow(laplaceImages[i], cmap="gray")
    imageCounter+=1
    plt.show()
    imageCounter=1
  plt_columns = 3
  for i in range(len(reconstructedImages)):
    plt.subplot(plt_rows, plt_columns,imageCounter)
    plt.title('Resize Gaus '+str(i))
    plt.imshow(reconstructedImages[i][0], cmap="gray")
    imageCounter+=1
    plt.subplot(plt_rows, plt_columns,imageCounter)
    plt.title('Resize Laplace '+str(i))
    plt.imshow(reconstructedImages[i][1], cmap="gray")
    imageCounter+=1
    plt.subplot(plt_rows, plt_columns,imageCounter)
    plt.title('Reconstructed '+str(i))
    plt.imshow(reconstructedImages[i][2], cmap="gray")
    imageCounter+=1
    plt.show()
    imageCounter=1
  
if __name__ == "__main__":
  main()

