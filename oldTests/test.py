import matplotlib.pyplot as plt
import cv2
import numpy as np

grid = np.zeros((50, 50), dtype=np.uint8)
cv2.line(grid, (25, 25), (-10, 20), (255, 255, 255), 1)
plt.imshow(grid)
plt.show()