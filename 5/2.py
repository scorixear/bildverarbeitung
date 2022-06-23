import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def loG(img):
  return kernel_transform(img, (1/16)*np.array([[0,1,2,1,0],[1,0,-2,0,1],[2,-2,-8,-2,2],[1,0,-2,0,1],[0,1,2,1,0]]))
def doG(img):
  return kernel_transform(img, np.array([[1,4,6,4,1],[4,0,-8,0,4],[6,-8,-28,-8,6],[4,0,-8,0,4],[1,4,6,4,1]]))
def laplace(img):
  return kernel_transform(img, np.array([[0,1,0],[1,-4,1],[0,1,0]]))

def sobel(img):
  gx = (1/8)*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  gy = (1/8)*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  imggx = kernel_transform(img, gx)
  imggy = kernel_transform(img, gy)
  return addImages([imggx, imggy])


def kernel_transform(img, kernel):
  return cv.filter2D(img, -1, kernel)

def addImages(images):
  for y in range(images[0].shape[0]):
    for x in range(images[0].shape[1]):
      val = 0
      for i in range(len(images)):
        val += images[i][y,x] * images[i][y,x]
      images[0][y,x] = int(np.sqrt(val))
  return images[0]


def findEdgePixel(img, x, y, threshold):
  for j in range(x, img.shape[1]):
    if img[y,j] > threshold:
      return j, y
  raise Exception("")

# start X index to search for edge
EDGE_FIND_X = 0
# start Y index to search for edge
EDGE_FIND_Y = 100
# Threshold of value when a pixel is considered an edge
EDGE_THRESHOLD = 200
# the Input Picture path
PICTURE_INPUT = "Rauschbild.png"
# Filtering Anything below EDGE_LOW_FILTER and anything above EDGE_HIGH_FILTER
EDGE_LOW_FILTER = 50
EDGE_HIGH_FILTER = 255
# if 8 neighbour is used
USE_8_NEIGHBOURS = False
# the edge detection filter method
FILTER_FUNCTION = sobel
INVERT_PICTURE= False
# use the Fill Method instead
USE_FILL_EDGE = False
# overwrite any edge detection filter and use original image instead
OVERWRITE_EDGE_DETECTION = False

# pixel class that keeps track of its visited neighbours
class Pixel:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    
    if USE_8_NEIGHBOURS:
      # eight side neighbours (Priority Order: Down Left, Down, Down Right, Right, Up Right, Up, Up Left, Left)
      self.neighbours = [(x-1, y+1), (x, y+1), (x+1,y+1), (x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y)]
    else:
      # four side neighbours (Priority Order: Down, Right, Up, Left)
      self.neighbours = [(x, y+1), (x+1, y), (x, y-1), (x-1, y)]

  def __eq__(self, o):
    return self.x == o.x and self.y == o.y
  def __hash__(self):
    return hash((self.x, self.y))
  def __repr__(self) -> str:
    return f"({self.x}, {self.y})"
  # returns the next neighbour in the priority order and removes it from "not visited" list
  def get_next_neighbour(self):
    if len(self.neighbours) > 0:
      pair = self.neighbours.pop(0)
      return Pixel(pair[0], pair[1])
    else:
      return None

# return true if the pixel is in the picture
def is_in_bounds(img, pixel):
  if pixel.x < 0 or pixel.x >= img.shape[1] or pixel.y < 0 or pixel.y >= img.shape[0]:
    return False
  return True

def fill_edges(img, x, y, threshold):
  painted_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      painted_img[i,j] = np.array([img[i,j], img[i,j], img[i,j]])
   # set start pixel to red
  stack = [Pixel(x,y)]
  edge = []
  while len(stack) > 0:
    pixel = stack.pop()
    if is_in_bounds(img, pixel) and img[pixel.y, pixel.x] > threshold and pixel not in edge:
      edge.append(pixel)
      for neighbour in pixel.neighbours:
        stack.append(Pixel(neighbour[0], neighbour[1]))
  for pixel in edge:
    painted_img[pixel.y, pixel.x] = np.array([255, 0, 0])
  return painted_img
  

def paint_edges(img, x, y, threshold):
  # create new image that is rgb and transform img to rgb
  painted_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      painted_img[i,j] = np.array([img[i,j], img[i,j], img[i,j]])
  # set start pixel to red
  painted_img[y,x] = np.array([255,0,0])
  # and add it to the edge list
  edges_visited = [Pixel(x,y)]
  # get the next pixel to be visited
  nextPixel = edges_visited[-1].get_next_neighbour()
  # used if we enter an infinit loop
  stepcounter = 0
  # used if we enter a dead end
  lastLength = 0
  noChangesCounter = 0
  # loop through next Pixel and its neighbours
  while True:
    # increase this every loop
    stepcounter += 1
    if stepcounter % 1000 == 0:
      # if we have a somewhat similar path length since the last 1000 checks, increase the noChangesCounter
      if abs(lastLength - len(edges_visited)) < 10:
        noChangesCounter+=1
      else:
        lastLength = len(edges_visited)
      # if we have a similar path length after 10.000 visits, exit the loop
      if noChangesCounter > 10:
        print ("Breaking due to no progress")
        break
      # output for debug
      print("Step:", stepcounter/1000000, "Path Length:", len(edges_visited))
    # if we somehow reached an infinit loop, break the loop
    if stepcounter > 1000000:
      print("Error: infinite loop")
      break
    
    # if no more neighbours can be visited
    if nextPixel is None:
      # pop the pixel from the list
      edges_visited.pop()
      # if we have no more pixels to visit, there is no loop to perform
      if len(edges_visited) == 0:
        break
      # backtrack to last pixel and get its next neighbour
      nextPixel = edges_visited[-1].get_next_neighbour()
      continue
    # if we completed the loop
    if nextPixel == edges_visited[0] and len(edges_visited) > 5:
      print("Completed Loop")
      break
    # if we have a neighbour and its position is valid
    if is_in_bounds(img, nextPixel):
      # if the neighbour is not visited yet
      if nextPixel not in edges_visited:
        # if the neighbour is above the threshold
        if img[nextPixel.y, nextPixel.x] > threshold:
          # add the neighbour to the list of visited pixels
          edges_visited.append(nextPixel)
    # get the next neighbour
    nextPixel = edges_visited[-1].get_next_neighbour()
  # after finishing the loop, paint the edges
  for edge in edges_visited:
    # paint the neighbour
    painted_img[edge.y, edge.x] = np.array([255,0,0])
  return painted_img

def main():
  # Read in picture
  img = plt.imread(os.path.join(os.path.dirname(__file__), PICTURE_INPUT))
  # if picture is rgb, convert to grayscale
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # if picture is in range between 0 and 1
  if img.max() <= 1:
    img = img*255
  # use Edge detection filter and cap the output to 0 and EDGE_HIGH_FILTER
  edge_img = np.clip(FILTER_FUNCTION(img),0,EDGE_HIGH_FILTER)
  # if the picture should be inverted, invert it
  if INVERT_PICTURE:
    edge_img = (~(np.array(edge_img, np.uint8)))
  # otherwise use a threshold between EDGE_LOW_FILTER and EDGE_HIGH_FILTER to clean up the edge picture
  else:
    edge_img = cv.threshold(edge_img, EDGE_LOW_FILTER, EDGE_HIGH_FILTER, cv.THRESH_BINARY)[1].astype(int)
  # plot the original and edge picture
  if OVERWRITE_EDGE_DETECTION:
    edge_img = img.copy()
  plt.subplot(231)
  plt.imshow(img, cmap="gray")
  plt.title("Original")
  plt.subplot(232)
  plt.imshow(edge_img, cmap="gray")
  plt.title(FILTER_FUNCTION.__name__)
  
  # start searching for edge pixel
  x,y = findEdgePixel(edge_img, EDGE_FIND_X, EDGE_FIND_Y, EDGE_THRESHOLD)
  # if edge pixel was found
  if x is not None and y is not None:
    # paint the edge pixels and show the result
    plt.subplot(233)
    painted_img = []
    if USE_FILL_EDGE:
      painted_img = fill_edges(edge_img, x, y, EDGE_THRESHOLD)
    else:
      painted_img = paint_edges(edge_img, x, y, EDGE_THRESHOLD)
    plt.imshow(painted_img)
    plt.title("Result")
  else:
    print("No edge found")
  plt.show()

if __name__ == "__main__":
  main()