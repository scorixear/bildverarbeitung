import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def imag_X(img, n, m=1):
  if n==1:
    tiled_img = img
  else:
    lst_imgs = []
    for i in range(n):
      lst_imgs.append(img)
    tiled_img = np.concatenate(lst_imgs, axis=1)
  if m > 1:
    lst_imgs = []
    for i in range(m):
      lst_imgs.append(tiled_img)
    tiled_img = np.concatenate(lst_imgs, axis=0)
  return tiled_img
basic_pattern = mpimg.imread('X.jpg')
decorators_img=imag_X(basic_pattern, 3, 4)
plt.axis("off")
plt.imshow(decorators_img)
plt.show()