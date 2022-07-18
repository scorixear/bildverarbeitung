# This scrict does the following:
#   1. Read in an image provided by the "PICTURE_INPUT" variable
#   2. Convert the image to grayscale (0-255)
#   3. Apply an edge filter to the image provided by the "FILTER_FUNCTION" variable
#   4. Applies a threshold to minimize the noise
#   5. Find the first edge pixel in the image
#   6. Traverses through the edge stepping from pixel to pixel following the Priority  Down Left, Down, Down Right, Right, Up Right, Up, Up Left, Left
#   7. Draws a line from the first edge pixel to the last edge pixel found
#   8. Converts the Edge to an edge track
#   9. Generates a gif from the edge track that paints the detected edge
#
# Notes:
#   - The filter function is essentiall to achive a good outcome
#   - For each image the best filter function and setting should be set - this can be hard or impossible
#   - The generated gif does not have any input from the original image but only the edge track

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2 as cv

# LoG filter - Edge pixels are around 70-100 brightness
# Result is with minimal noise but large gaps in the edge
def loG(img):
  return kernel_transform(img, (1/16)*np.array([[0,1,2,1,0],[1,0,-2,0,1],[2,-2,-8,-2,2],[1,0,-2,0,1],[0,1,2,1,0]]))
# DoG filter - Edge pixels are around 255 brightness
# Results are with maximal noise but no gaps in the edge
def doG(img):
  return kernel_transform(img, np.array([[1,4,6,4,1],[4,0,-8,0,4],[6,-8,-28,-8,6],[4,0,-8,0,4],[1,4,6,4,1]]))
# Laplace filter - Edge pixels are around 30 - 120 brightness
# Results are with medium noise and small gaps in the edge
def laplace(img):
  return kernel_transform(img, np.array([[0,1,0],[1,-4,1],[0,1,0]]))
# Sobel filter - Edge pixels are around 100 brightneses
# Results are with minimal noise and no gaps in the edge
def sobel(img):
  gx = (1/8)*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  gy = (1/8)*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  imggx = kernel_transform(img, gx)
  imggy = kernel_transform(img, gy)
  return addImages([imggx, imggy])

# Used to apply a kernel to an image via cv2
def kernel_transform(img, kernel):
  return cv.filter2D(img, -1, kernel)

# Returns the Sum of two images scaled by their square root
def addImages(images):
  for y in range(images[0].shape[0]):
    for x in range(images[0].shape[1]):
      val = 0
      for i in range(len(images)):
        val += images[i][y,x] * images[i][y,x]
      images[0][y,x] = int(np.sqrt(val))
  return images[0]

# Returns the first bright pixel in one row
def findEdgePixel(img, x, y, threshold):
  for j in range(x, img.shape[1]):
    if img[y,j] > threshold:
      return j, y
  raise Exception("No Edge Pixel found")

#start X index to search for edge
EDGE_FIND_X = 0
# start Y index to search for edge
EDGE_FIND_Y = 100
# Threshold of value when a pixel is considered an edge
EDGE_THRESHOLD = 100
# the Input Picture path
PICTURE_INPUT = "Rauschbild_clean.png"
# Filtering Anything below EDGE_LOW_FILTER and anything above EDGE_HIGH_FILTER
EDGE_LOW_FILTER = 100
EDGE_HIGH_FILTER = 255
# if 8 neighbour is used
USE_8_NEIGHBOURS = True
# the edge detection filter method
FILTER_FUNCTION = laplace
INVERT_PICTURE= False
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
  # used if we enter an infinite loop (above 100.000 steps)
  stepcounter = 0
  # used if we enter a dead end (Length of the last calculated edge - set every 100 steps)
  lastLength = 0
  # counter with similar lengths detected (backtracking just takes too long)
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
    elif is_in_bounds(img, nextPixel):
      # if the neighbour is not visited yet
      if nextPixel not in edges_visited:
        # if the neighbour is above the threshold
        if img[nextPixel.y, nextPixel.x] > threshold:
          # add the neighbour to the list of visited pixels
          edges_visited.append(nextPixel)
    # get the next neighbour
    nextPixel = edges_visited[-1].get_next_neighbour()
  # after finishing the loop, paint the edges

  # the edge track containes pairs of (x,y) coordinates and their direction (0 = right, 1 = up right, 2 = up, 3 = up left, 4 = left, 5 = down left, 6 = down, 7 = down right)
  edgeTrack = []
  # loop through all edge pixels
  for edge in edges_visited:
    # add the corresponding edge to the edge track
    addEdgeTrack(edgeTrack, edge)
    # paint the edge in the final image red
    painted_img[edge.y, edge.x] = np.array([255,0,0])
  return painted_img, edgeTrack

# adds the edge to the edge track with its corresponding direction
def addEdgeTrack(track, point):
  # if this is the first point of the edge track
  if len(track)== 0:
    track.append((point, 0))
    return
  # retrieve previous point
  lastPoint = track[-1][0]
  directions = [(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)]
  # loop through all directions
  for index, direction in enumerate(directions):
    # if last point + direction is current point, we found the direction
    if (lastPoint.x + direction[0] == point.x and lastPoint.y + direction[1] == point.y):
      track.append((point, index))
      return
  # exception - provided point cannot be reached from previous point - disruptive edge
  raise Exception("No matching Direction found", lastPoint, point)

# generates frame pictures for each next step of the edge track
def generateFrame(track, index, start, shape):
  img = np.zeros(shape).astype(int)
  img[start[1], start[0]] = 255
  for i in range(index+1):
    point = track[i][0]
    img[point.y, point.x] = 255
  return [plt.imshow(img, cmap="gray", animated=True)]

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
  edge_track = []
  if x is not None and y is not None:
    # paint the edge pixels and show the result
    plt.subplot(233)
    painted_img, edge_track = paint_edges(edge_img, x, y, EDGE_THRESHOLD)
    
    # generate edge_string for the edge track
    edge_string = ""
    for pixel in edge_track:
      edge_string += str(pixel[1]) + ""
    # and print it
    print(f"({x},{y})",edge_string)
    plt.imshow(painted_img)
    plt.title("Result")
    # generate gif for the edge track
    fig = plt.figure()
    frames = []
    # loop through all steps of the edge track
    for index in range(len(edge_track)):
      # generate frame for the current step
       frames.append(generateFrame(edge_track, index, (x,y), img.shape))
    # save the gif
    ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True, repeat_delay=0)
    ani.save("animated.gif", fps=120)
  else:
    print("No edge found")
  plt.show()

if __name__ == "__main__":
  main()