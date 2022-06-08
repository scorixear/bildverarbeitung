import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Returns Translation Matrix
def get_translation_matrix(x, y):
  return np.array([
    [1,0,x],
    [0,1,y],
    [0,0,1]])

# Returns Rotation Matrix
def getRotationMatrix(phi):
  # get radiant of phi
  phi = np.deg2rad(phi)
  return np.array([
    [np.cos(phi), -np.sin(phi),0],
    [np.sin(phi), np.cos(phi),0],
    [0,0,1]])

# Interpolates closest pixels
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

# Returns color of nearest neighbour
def nearestNeighbour(img, x, y):
  newX = int(round(x))
  newY = int(round(y))
  return img[newY, newX]

# Returns invers matrix multiplication
def matrix_mult():
  # collect matricies in correct order
  matricies = [getRotationMatrix(21), get_translation_matrix(30, 25)]
  # start multiplying martrices
  multiplied = matricies[0]
  for i in range(1, len(matricies)):
    # if only one matrix was added, skip this step
    if(len(matricies) < i):
      break
    # otherwise matrix mulitply the product with the next matrix
    multiplied = np.matmul(multiplied, matricies[i])
  # inverse the final matrix
  return np.linalg.inv(multiplied)

# Applies transformation matrix to image
def transformImage(img: np.ndarray):
  # get invers transformation matrix
  transformationMatrix = matrix_mult()
  # get new image filled with white pixels
  newimg = np.ones((img.shape[0], img.shape[1], 3)).astype(int)*255
  # iterate over each pixel
  for newY in range(img.shape[0]):
    for newX in range(img.shape[1]):
      # calculate original image position
      origPos=np.matmul(transformationMatrix, np.array([[newX],[newY],[1]]))
      # if original image position exceedes image boundaries, skip the pixel (leave it white)
      if origPos[1,0]<0 or origPos[1,0]>=newimg.shape[0]-1 or origPos[0,0]<0 or origPos[0,0]>=newimg.shape[1]-1:
        continue
      # otherwise, interpolate color from original image
      newimg[newY,newX]=bilinearInterpolate(img, origPos[0,0], origPos[1,0])
  # draw results
  plt.subplot(121)
  plt.imshow(img)
  plt.axis('off')
  plt.title('Original Image')
  plt.subplot(122)
  plt.imshow(newimg)
  plt.axis('off')
  plt.title('Transformed Image')



def main():
  # read in image
  img = plt.imread(os.path.join(os.path.dirname(__file__),"cutecat.jpg"))
  transformImage(img)
  plt.show()

if __name__ == "__main__":
  sys.exit(main())
