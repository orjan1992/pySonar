import matplotlib.pyplot as plt
from ogrid.oGrid import *
import numpy as np

grid = OGrid(False, 0.5, 0.7)
# grid.oLog[0:100, 0:100] = np.ones(np.shape(grid.oLog[0:100, 0:100]))*10
# grid.oLog[500:600, 600:800] = np.ones(np.shape(grid.oLog[500:600, 600:800]))*2
# grid.oLog[700:800, 1000:1200] = np.ones(np.shape(grid.oLog[700:800, 1000:1200]))*3
# grid.oLog[300:400, 50:200] = np.ones(np.shape(grid.oLog[300:400, 50:200]))*4

grid.oLog = np.random.random(np.shape(grid.oLog))
obst = grid.get_obstacles()

plt.subplot(211)
plt.imshow(obst.T)

plt.subplot(212)
plt.imshow(grid.get_p().T)
plt.show()
