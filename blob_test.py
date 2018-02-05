import matplotlib.pyplot as plt
from ogrid.oGrid import *
import numpy as np

grid = OGrid(False, 0.5, 0.7)
# grid.o_log[0:100, 0:100] = np.ones(np.shape(grid.o_log[0:100, 0:100]))*10
# grid.o_log[500:600, 600:800] = np.ones(np.shape(grid.o_log[500:600, 600:800]))*2
# grid.o_log[700:800, 1000:1200] = np.ones(np.shape(grid.o_log[700:800, 1000:1200]))*3
# grid.o_log[300:400, 50:200] = np.ones(np.shape(grid.o_log[300:400, 50:200]))*4

grid.o_log = np.random.random(np.shape(grid.o_log))
obst = grid.get_obstacles()

plt.subplot(211)
plt.imshow(obst.T)

plt.subplot(212)
plt.imshow(grid.get_p().T)
plt.show()
