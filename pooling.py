import numpy as np
import cv2
from scipy import misc

import matplotlib.pyplot as plt

i = misc.ascent()


copy = np.copy(i)

x = copy.shape[0]
y = copy.shape[1]

new_x = int(x/2)
new_y = int(y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, x, 2):
  for y in range(0, y, 2):
    pixels = []
    pixels.append(copy[x, y])
    pixels.append(copy[x+1, y])
    pixels.append(copy[x, y+1])
    pixels.append(copy[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]
 
# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()