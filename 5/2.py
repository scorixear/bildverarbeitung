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

def kernel_transform(img, kernel):
  return cv.filter2D(img, -1, kernel)

def findEdgePixel(img, x, y, threshold):
  for j in range(x, img.shape[1]):
    if img[y,j] > threshold:
      return j, y
  return None, None

EDGE_FIND_X = 20
EDGE_FIND_Y = 100
EDGE_THRESHOLD = 200
PICTURE_INPUT = "Rauschbild.png"
EDGE_LOW_FILTER = 254
EDGE_HIGH_FILTER = 255
USE_8_NEIGHBOURS = False
FILTER_FUNCTION = doG
INVERT_PICTURE= False

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
  def get_next_neighbour(self):
    if len(self.neighbours) > 0:
      pair = self.neighbours.pop(0)
      return Pixel(pair[0], pair[1])
    else:
      return None

def is_in_bounds(img, pixel):
  if pixel.x < 0 or pixel.x >= img.shape[1] or pixel.y < 0 or pixel.y >= img.shape[0]:
    return False
  return True

def paint_edges(img, x, y, threshold):
  painted_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      painted_img[i,j] = np.array([img[i,j], img[i,j], img[i,j]])
  
  painted_img[y,x] = np.array([255,0,0])
  edges_visited = [Pixel(x,y)]
  nextPixel = edges_visited[-1].get_next_neighbour()
  stepcounter = 0
  lastLength = 0
  noChangesCounter = 0
  while True:
    stepcounter += 1
    if stepcounter % 1000 == 0:
      if abs(lastLength - len(edges_visited)) < 10:
        noChangesCounter+=1
      else:
        lastLength = len(edges_visited)
      if noChangesCounter > 10:
        print ("Breaking due to no progress")
        break
      print("Step:", stepcounter/1000000, "Path Length:", len(edges_visited))
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
  for edge in edges_visited:
    # paint the neighbour
    painted_img[edge.y, edge.x] = np.array([255,0,0])
  return painted_img

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), PICTURE_INPUT))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  img = img*255
  edge_img = np.clip(FILTER_FUNCTION(img),0,EDGE_HIGH_FILTER)
  if INVERT_PICTURE:
    edge_img = (~(np.array(edge_img, np.uint8)))
  else:
    edge_img = cv.threshold(edge_img, EDGE_LOW_FILTER, EDGE_HIGH_FILTER, cv.THRESH_BINARY)[1].astype(int)
  #edge_img = img.copy().astype(int)
  plt.subplot(231)
  plt.imshow(img, cmap="gray")
  plt.title("Orig")
  plt.subplot(232)
  plt.imshow(edge_img, cmap="gray")
  plt.title("LoG")
  
  
  x,y = findEdgePixel(edge_img, EDGE_FIND_X, EDGE_FIND_Y, EDGE_THRESHOLD)
  if x is not None and y is not None:
    plt.subplot(233)
    painted_img = paint_edges(edge_img, x, y, EDGE_THRESHOLD)
    plt.imshow(painted_img)
    plt.title("Edge Detected")
  else:
    print("No edge found")
  plt.show()

if __name__ == "__main__":
  main()