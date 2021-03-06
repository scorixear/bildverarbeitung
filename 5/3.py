from calendar import c
import os
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sqlalchemy import false

# start X index to search for edge
SEED_X = 70
# start Y index to search for edge
SEED_Y = 100

EDGE_X = 20
EDGE_Y = 100
# Threshold of value when a pixel is considered an edge
THRESHOLD = 70
# the Input Picture path
PICTURE_INPUT = "Rauschbild.png"

USE_BOUNDARY_FILL = True

def is_in_bounds(img, pixel):
  if pixel[0] < 0 or pixel[0] >= img.shape[1] or pixel[1] < 0 or pixel[1] >= img.shape[0]:
    return False
  return True

def boundary_fill(img, seed, edge_seed, edge_threshold):
  painted_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      painted_img[i,j] = np.array([img[i,j], img[i,j], img[i,j]])
  stack = [edge_seed]
  edge = []
  while len(stack) > 0:
    pixel = stack.pop()
    if is_in_bounds(img, pixel) and img[pixel[1], pixel[0]] < edge_threshold and pixel not in edge:
      edge.append(pixel)
      for x in range(pixel[0]-1, pixel[0]+2):
        for y in range(pixel[1]-1, pixel[1]+2):
          if x == pixel[0] and y == pixel[1]:
            continue
          stack.append((x, y))
  region = []
  region_stack = [seed]
  while len(region_stack)> 0:
    pixel = region_stack.pop()
    if is_in_bounds(img, pixel) and pixel not in edge and pixel not in region:
      region.append(pixel)
      for x in range(pixel[0]-1, pixel[0]+2):
        for y in range(pixel[1]-1, pixel[1]+2):
          if x == pixel[0] and y == pixel[1]:
            continue
          region_stack.append((x, y))
  for r in region:
    painted_img[r[1], r[0]] = np.array([255,0,0])
  for e in edge:
    painted_img[e[1], e[0]] = np.array([0,0,255])
  return painted_img



def paint_area(img, seed, threshold):
  # create a copy of the image in rgb
  painted_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      painted_img[i,j] = np.array([img[i,j], img[i,j], img[i,j]])
  # used for saving the region
  region = [seed]
  # stack for visiting neighbours
  visited = [seed]
  print("Finding region")
  while len(visited) > 0:
    x,y = visited.pop()
    painted_img[y,x] = np.array([255,0,0])
    for i in range(y-1, y+2):
      for j in range(x-1, x+2):
        if i >= 0 and i < img.shape[0] and j >= 0 and j < img.shape[1]:
          if img[i,j] > threshold:
            if not np.array_equal(painted_img[i,j], np.array([255,0,0])):
              painted_img[i,j] = np.array([255,0,0])
              if (j, i) not in  region:
                  region.append((j, i))
              visited.append((j,i))
  print("Finding region edges")
  regionEdge = []
  for r in region:
    found_edge = False
    for i in range(r[0]-1, r[0]+2):
      for j in range(r[1]-1, r[1]+2):
        if i == r[0] and j == r[1]:
          continue
        if (i,j) not in region:
          found_edge = True
          break
      if found_edge:
        break
    if found_edge:
      regionEdge.append(r)
  print("Printing Edge Region")
  for r in regionEdge:
    painted_img[r[1], r[0]] = np.array([0,0,255])
  return painted_img


def main():
  # read in image
  img = plt.imread(os.path.join(os.path.dirname(__file__), PICTURE_INPUT))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if img.max() <= 1:
    img = img*255
  plt.subplot(221)
  plt.imshow(img, cmap="gray")
  plt.title("Original")
  plt.subplot(222)
  if USE_BOUNDARY_FILL:
    plt.imshow(boundary_fill(img, (SEED_X,SEED_Y), (EDGE_X, EDGE_Y), THRESHOLD).astype(int))
  else:
    plt.imshow(paint_area(img, (SEED_X,SEED_Y), THRESHOLD).astype(int))
  plt.title("Painted")
  plt.show()


if __name__ == "__main__":
  main()