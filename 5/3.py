import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def kernel_transform(img, kernel):
  return cv.filter2D(img, -1, kernel)

def loG(img):
  return kernel_transform(img, (1/16)*np.array([[0,1,2,1,0],[1,0,-2,0,1],[2,-2,-8,-2,2],[1,0,-2,0,1],[0,1,2,1,0]]))
def doG(img):
  return kernel_transform(img, np.array([[1,4,6,4,1],[4,0,-8,0,4],[6,-8,-28,-8,6],[4,0,-8,0,4],[1,4,6,4,1]]))
def laplace(img):
  return kernel_transform(img, np.array([[0,1,0],[1,-4,1],[0,1,0]]))

def paint_area(img, seed, threshold):
  painted_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      painted_img[i,j] = np.array([img[i,j], img[i,j], img[i,j]])
  visited = [seed]

  while len(visited) > 0:
    x,y = visited.pop()
    painted_img[y,x] = np.array([255,0,0])
    for i in range(y-1, y+2):
      for j in range(x-1, x+2):
        if i >= 0 and i < img.shape[0] and j >= 0 and j < img.shape[1]:
          if img[i,j] > threshold:
            if not np.array_equal(painted_img[i,j], np.array([255,0,0])):
              painted_img[i,j] = np.array([255,0,0])
              visited.append((j,i))
  return painted_img


def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'Rauschbild_clean.png'))
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  img = img*255
  plt.subplot(221)
  plt.imshow(img, cmap="gray")
  plt.title("Orig")
  plt.subplot(222)
  plt.imshow(paint_area(img, (75,100), 70).astype(int))
  plt.title("Painted")
  plt.show()


if __name__ == "__main__":
  main()