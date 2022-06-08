import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('test.jpg')
data = np.asarray(img, dtype='uint8')
imgpot = plt.imshow(data)
plt.show()