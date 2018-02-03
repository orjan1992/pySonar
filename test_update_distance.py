from ogrid.oGrid import OGrid
from matplotlib import pyplot as plt
grid = OGrid(False, 0.5)
grid.oLog[200:1400, 200:1400] = 1
plt.imshow(grid.oLog.T)
plt.show()
grid.current_distance = 15
grid.update_distance(5)
plt.imshow(grid.oLog.T)
plt.show()
