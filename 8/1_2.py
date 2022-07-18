import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2 as cv

START_X = 26
START_Y = 100

def cv_kernel(img, matrix):
  matrix[matrix == 0] = -1
  matrix[matrix == None] = 0
  return cv.morphologyEx(img, cv.MORPH_HITMISS, matrix.astype(int))

def thin(img):
  newimg = cv.bitwise_and(img,    cv.bitwise_not(cv_kernel(img, np.array([[0,None,1],[0,1,1],[0,None,1]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[0,0,None],[0,1,1],[None,1,1]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[0,0,0],[None,1,None],[1,1,1]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[None,0,0],[1,1,0],[1,1,None]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,None,0],[1,1,0],[1,None,0]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,1,None],[1,1,0],[None,0,0]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,1,1],[None,1,None],[0,0,0]]))))
  newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[None,1,1],[0,1,1],[0,0,None]]))))
  return newimg

def walkEdge(img):
  edge_track = []
  painted_img = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      if img[y,x] == 1:
        painted_img[y,x] = np.array([255,255,255])
  edge_track.append(((START_X,START_Y), 0))
  edges_visited = [edge_track[0][0]]
  directions = [(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)]
  iterations = 0
  while True:
    iterations+=1
    if iterations > 100000:
      print("Infinite Loop")
      break
    if iterations%100 == 0:
      print("Edges visited:", iterations)
    nextPixel = None
    direction = None
    for index, direction in enumerate(directions):
      nextPixel = (edges_visited[-1][0] + direction[0], edges_visited[-1][1] + direction[1])
      if nextPixel[0] < 0 or nextPixel[0] >= img.shape[0] or nextPixel[1] < 0 or nextPixel[1] >= img.shape[1]:
        nextPixel = None
        direction = None
        continue
      if img[nextPixel[0], nextPixel[1]] == 0 or nextPixel in edges_visited:
        nextPixel = None
        direction = None
        continue
      direction = index
      break
    if direction is None:
      print("Reached dead end")
      break
    edges_visited.append(nextPixel)
    edge_track.append((nextPixel, direction))
    painted_img[nextPixel[0], nextPixel[1]] = np.array([255,0,0])
  return painted_img, edge_track
    
def generateFrame(track, index, start, shape):
  img = np.zeros(shape).astype(int)
  img[start[1], start[0]] = 255
  for i in range(index+1):
    point = track[i][0]
    img[point.y, point.x] = 255
  return [plt.imshow(img, cmap="gray", animated=True)]

def main():
  img = plt.imread(os.path.join(os.path.dirname(__file__), 'skeleton.png'))
  if img.ndim == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if img.max() <= 1:
    img = img * 255
  img = img.astype('uint8')

  binary = np.where(img > 127, 1, 0).astype('uint8')
  # mpimg.imsave(os.path.join(os.path.dirname(__file__), 'binary1.png'), binary, cmap='gray')
  #binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3,3), dtype='uint8'), iterations=3)
  #binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((3,3), dtype='uint8'), iterations=3)
  oldPic = binary.copy()
  
  
  thinned = thin(oldPic)
  zeros = cv.countNonZero(thinned)
  prevZeros = cv.countNonZero(oldPic)
  while zeros != 0 and zeros != prevZeros:
    print("Recalculating thinning",zeros)
    oldPic = thinned
    thinned = thin(oldPic)
    prevZeros = zeros
    zeros = cv.countNonZero(thinned)
  
  painted_img, edge_track = walkEdge(oldPic)
  plt.subplot(231)
  plt.imshow(binary, cmap="gray")
  plt.title("Original")
  plt.subplot(232)
  plt.imshow(oldPic, cmap="gray")
  plt.title("Thinned")
  plt.subplot(233)
  plt.title("Edge Painted")
  plt.imshow(painted_img)
  fig = plt.figure()
  frames = []
  edge_track.pop(0)
  for index in range(len(edge_track)):
      frames.append(generateFrame(edge_track, index, (START_X, START_Y), img.shape))
  ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True, repeat_delay=0)
  plt.show()
  
if __name__ == "__main__":
  main()
    