from ogrid.oGrid import OGrid
from matplotlib import pyplot as plt
import numpy as np
grid = OGrid(False, 0.5)
grid.o_log[200:1400, 200:1400] = np.random.random(np.shape(grid.o_log[200:1400, 200:1400]))
plt.imshow(grid.o_log.T)
plt.show()
grid.last_distance = 5
grid.update_distance(7)
plt.imshow(grid.o_log.T)
plt.show()

#
# import numpy as np
# from scipy.interpolate import *
# from matplotlib import pyplot as plt
#
# last_distance = 15
# distance = 10
#
# x = np.linspace(0, 10, 11, True)/10.0
# xy_mesh_unit = np.meshgrid(x, x)
# o_log = np.random.random((11, 11))
#
# old_x, old_y = np.multiply(xy_mesh_unit, last_distance)
# new_xy = x*distance
# tmp = np.array([old_x.ravel(), old_y.ravel()]).T
# new_grid = griddata(tmp, o_log.ravel(), (new_xy,new_xy), method='linear', fill_value=5)
# # zfun_smooth_rbf = Rbf(old_xy, old_xy, o_log, function='linear', smooth=0)  # default s
# # z_dense_smooth_rbf = zfun_smooth_rbf(new_xy, new_xy)
# plt.contourf(old_xy, old_xy, o_log)
# plt.show()