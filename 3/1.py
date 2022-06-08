import sys, os
import numpy as np
import matplotlib.pyplot as plt

def mask(xmin, xmax, ymin, ymax, img):
  for i in range(xmin, xmax):
    for j in range(ymin, ymax):
      img[j,i] = 0
def invertedMask(xmin, xmax, ymin, ymax, img):
  for i in range(len(img)):
    for j in range(len(img[i])):
      if not (i >= xmin and i <= xmax and j >= ymin and j <= ymax):
        img[i,j] = 0

def removeBlanks(img, size):
  #mask(0, int(img.shape[1]/2)-size, 0, int(img.shape[0]/2)-size, img)
  #mask(0, int(img.shape[1]/2)-size, int(img.shape[0]/2)+size, img.shape[0], img)
  #mask(int(img.shape[1]/2)+size, img.shape[1], 0, int(img.shape[0]/2)-size, img)
  #mask(int(img.shape[1]/2)+size, img.shape[1], int(img.shape[0]/2)+size, img.shape[0], img)
  #mask(int(img.shape[1]/2)-size, int(img.shape[1]/2)+size, 0, img.shape[0], img)
  #mask(0, img.shape[1], int(img.shape[0]/2)-size, int(img.shape[0]/2)+size, img)
  #invertedMask(int(len(img)/2)-size, int(len(img)/2)+size, int(len(img[0])/2)-size, int(len(img[0])/2)+size, img)
  halfx = int(img.shape[1]/2)
  halfy = int(img.shape[0]/2)
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      if (x < halfx - size*3) or ( x > halfx + size*3) or ( y < halfy - size*3) or ( y > halfy + size*3) or (x < halfx - size and y < halfy - size) or (x > halfx + size and y < halfy - size) or (x < halfx - size and y > halfy + size) or (x > halfx + size and y > halfy + size):
        if np.log(abs(img[y,x])) > 4.5:
          img[y,x] = img[y,x]*0.1
        elif np.log(abs(img[y,x])) > 2.5:
          img[y,x] = img[y,x]*0.5


def a(img):
  fftimg = np.fft.fftshift(np.fft.fft2(img))
  plt.subplot(221)
  plt.imshow(img, cmap=plt.get_cmap('gray'))
  plt.subplot(222)
  plt.imshow(np.log(abs(fftimg)), cmap='gray')

  removeBlanks(fftimg, 50)
  plt.subplot(223)
  plt.imshow(np.log(abs(fftimg)), cmap='gray')
  plt.subplot(224)
  plt.imshow(np.log(abs(np.fft.ifft2(fftimg))),cmap='gray')

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__),"moonlanding.png"))
  a(img)
  plt.show()

if __name__ == "__main__":
  sys.exit(main())