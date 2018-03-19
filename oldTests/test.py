import numpy as np
from matplotlib import pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from messages.sonarMsg import SonarMsg
from ogrid.rawGrid import RawGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.readLogFile import ReadLogFile

csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    log = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
    file = ReadCsvFile(log, 4002, 13102, cont_reading=False)
else:
    log = '/home/orjangr/Repos/pySonar/logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = RawGrid(0.2, 20, 10, 0.5)
Threshold = 60
theta = np.zeros(1)
while file.messagesReturned < 10:
    msg = file.read_next_msg()
    if type(msg) is SonarMsg and msg.type == 2:
        # print(file.messagesReturned)
        O.auto_update_zhou(msg, Threshold)
        theta = np.append(theta, msg.bearing)
    elif msg == -1:
        break
file.close()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(O.cell_x_value, O.cell_y_value, np.arcsin((O.cellSize+O.cell_x_value)/O.r_unit)-O.theta,
#                        cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
#
# # plt(O.cell_x_value, O.cell_y_value, np.arcsin((O.cellSize+O.cell_x_value)/O.r_unit)-O.theta)
# # plt.plot(O.cell_x_value[0, :], (O.cellSize+O.cell_x_value[0, :])/np.max(np.max(O.r_unit)))
# plt.show()
delta_psi = 1*pi/180
delta_x = O.r*np.sin(delta_psi+O.theta) - O.cell_x_value
delta_y = O.r*np.cos(delta_psi+O.theta) - O.cell_y_value
xmax = np.max(np.max(np.abs(delta_x)))
ymax = np.max(np.max(np.abs(delta_y)))
print(xmax)
print(ymax)
print(np.nonzero(np.abs(delta_x)==xmax))
print(np.nonzero(np.abs(delta_y)==ymax))
print(O.jMax)