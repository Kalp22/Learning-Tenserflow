import numpy as np
import cv2
from scipy import misc

import matplotlib.pyplot as plt

i = misc.ascent()


copy = np.copy(i)

x = copy.shape[0]
y = copy.shape[1]


filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]

weight = 1

for x in range(1,x-1):
  for y in range(1,y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (i[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (i[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (i[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (i[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (i[x, y] * filter[1][1])
      output_pixel = output_pixel + (i[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (i[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (i[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (i[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      copy[x, y] = output_pixel

plt.gray()
plt.grid(False)
plt.axis('off')
plt.imshow(copy)
plt.show()