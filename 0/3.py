import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

basic_pattern = mpimg.imread('X.jpg')
cropped = basic_pattern[0:200, 0:120]
plt.axis("off")
plt.imshow(cropped);
plt.show()