import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def bilinearInterpolate(img, x, y):
  # get coordinates of 4 closest pixels
  x2 = int(x+1)
  x1 = int(x)
  y2 = int(y+1)
  y1 = int(y)

  # get colors of 4 closest pixels
  q11 = img[y1,x1]
  q12 = img[y2,x1]
  q21 = img[y1,x2]
  q22 = img[y2,x2]

  # interpolate colors for top two pixels
  fx_y1=(x2-x)/(x2-x1)*q11+(x-x1)/(x2-x1)*q21
  # interpolate color for bottom two pixels
  fx_y2=(x2-x)/(x2-x1)*q12+(x-x1)/(x2-x1)*q22
  # interpolate between results
  return (y2-y)/(y2-y1)*fx_y1+(y-y1)/(y2-y1)*fx_y2

def scale(img: np.ndarray, percent):
  if percent == 100:
    return img
  newimg = np.ones((int(img.shape[0]*(percent/100)), int(img.shape[1]*(percent/100))))*255
  matrix = np.linalg.inv(np.array([
    [percent/100,0,0],
    [0,percent/100, 0],
    [0,0,1]
  ]))
  for y in range(newimg.shape[0]):
    for x in range(newimg.shape[1]):
      origCoord = np.matmul(matrix, np.array([[x],[y],[1]]))
      if origCoord[0,0] < 0 or origCoord[0,0]>=(img.shape[1]-1) or origCoord[1,0] < 0 or origCoord[1,0]>=(img.shape[0]-1):
        continue
      newimg[y,x]=bilinearInterpolate(img,origCoord[0,0],origCoord[1,0])
  return newimg

def fourier_transform(img, function):
  
  fftimg = np.fft.fft2(img)
  fftimg = np.fft.fftshift(fftimg)
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      fftimg[y,x] = function(((x*2)/img.shape[1])-1, ((y*2)/img.shape[0])-1)
  return np.log(abs(np.fft.ifft2(fftimg)))

def addImages(images):
  for y in range(images[0].shape[0]):
    for x in range(images[0].shape[1]):
      val = 0
      for i in range(len(images)):
        val += images[i][y,x] * images[i][y,x]
      images[0][y,x] = int(np.sqrt(val))
  return images[0]

def kernel_transform(img, kernel):
  return cv.filter2D(img, -1, kernel)

def laplace(img):
  return kernel_transform(img, np.array([[0,1,0],[1,-4,1],[0,1,0]]))
  #return fourier_transform(img, lambda x,y: (-4)*(np.sin((np.pi*x)/2)**2)-4*(np.sin((np.pi*y)/2)**2))

def sobel(img):
  gx = (1/8)*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  gy = (1/8)*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  imggx = kernel_transform(img, gx)
  imggy = kernel_transform(img, gy)
  return addImages([imggx, imggy])

def cannyEdge(img):
  #gx = fourier_transform(img, lambda x,y: complex(0,1)*np.sin(np.pi*x)*(np.cos((np.pi*x)/2)**2)*(np.cos((np.pi*y)/2)**2))
  gx = kernel_transform(img, (1/32)*np.array([[1,2,0,-2,-1],[2,4,0,-4,-2],[1,2,0,-2,-1]]))
  #gy = fourier_transform(img, lambda x,y: complex(0,1)*np.sin(np.pi*y)*(np.cos((np.pi*x)/2)**2)*(np.cos((np.pi*y)/2)**2))
  gy = kernel_transform(img, (1/32)*np.array([[1,2,1],[2,4,2],[0,0,0],[-2,-4,-2],[-1,-2,-1]]))
  return addImages([gx, gy])
  #return cv.Canny(img, 100, 200)

def loG(img):
  return kernel_transform(img, (1/16)*np.array([[0,1,2,1,0],[1,0,-2,0,1],[2,-2,-8,-2,2],[1,0,-2,0,1],[0,1,2,1,0]]))
  #return fourier_transform(img, lambda x,y: 4*((np.sin((np.pi*x)/2)**2)+(np.sin((np.pi*y)/2)**2))*(np.cos((np.pi*x)/2)**2)*(np.cos((np.pi*y)/2)**2))
def doG(img):
  return kernel_transform(img, np.array([[1,4,6,4,1],[4,0,-8,0,4],[6,-8,-28,-8,6],[4,0,-8,0,4],[1,4,6,4,1]]))
  #return fourier_transform(img, lambda x,y: 4*(np.cos((np.pi*x)/2)**4)*(np.cos((np.pi*y)/2)**4)-4*(np.cos((np.pi*x)/2)**2)*(np.cos((np.pi*y)/2)**2))

def showImages(img, cvimg):
  plt.subplot(231)
  plt.imshow(img, cmap="gray")
  plt.title("Orig")
  plt.subplot(232)
  plt.imshow(laplace(img), cmap="gray")
  plt.title("Laplace")
  plt.subplot(233)
  plt.imshow(sobel(img), cmap="gray")
  plt.title("Sobel")
  plt.subplot(234)
  plt.imshow(cannyEdge(cvimg), cmap="gray")
  plt.title("Canny-Edge")
  plt.subplot(235)
  plt.imshow(loG(img), cmap="gray")
  plt.title("LoG")
  plt.subplot(236)
  plt.imshow(doG(img), cmap="gray")
  plt.title("DoG")
  plt.show()

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__),"../5/Rauschbild.png"))
  # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
  img50 = cv.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv.INTER_AREA)
  img200 = cv.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)), interpolation=cv.INTER_AREA)

  cvimg = cv.imread(os.path.join(os.path.dirname(__file__),"../5/Rauschbild.png"),0)
  cvimg50 = cv.resize(cvimg, (int(cvimg.shape[1]*0.5), int(cvimg.shape[0]*0.5)), interpolation=cv.INTER_AREA)
  cvimg200 = cv.resize(cvimg, (cvimg.shape[1]*2, cvimg.shape[0]*2), interpolation=cv.INTER_AREA)

  showImages(img, cvimg)
  showImages(img50, cvimg50)
  showImages(img200, cvimg200)

if __name__ == "__main__":
  main()