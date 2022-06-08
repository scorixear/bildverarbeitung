import matplotlib.pyplot as plt
import numpy as np
an_image = plt.imread('test.jpg');
#plt.imshow(an_image)
weights = [0, 0, 1]

# an_image[...,:3] is returning the red green and blue values of each pixel 
# np.dot multiplies the weights with matrix multiplication with each rgb value
# the return is a grey picture
grayscale_image = np.dot(an_image[...,:3], weights)
plt.imshow(grayscale_image, cmap=plt.get_cmap("gray"))
plt.show();