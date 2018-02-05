import numpy as np
from matplotlib import pyplot as plt
from math import pi

from messages.sonarMsg import SonarMsg
from ogrid.oGrid import OGrid
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

O = OGrid(0.2, 20, 10, 0.5)
O2 = OGrid(0.2, 20, 10, 0.5)
Threshold = 60
theta = np.zeros(1)
while file.messagesReturned < 10:
    msg = file.read_next_msg()
    if type(msg) is SonarMsg and msg.type == 2:
        print(file.messagesReturned)
        O.auto_update_zhou(msg, Threshold)
        O2.auto_update_zhou(msg, Threshold)
        theta = np.append(theta, msg.bearing)
    elif msg == -1:
        break
file.close()
ax = plt.subplot(131)
ax.set(xlabel='X [m]', ylabel='Y [m])')
img = ax.imshow(O.get_p(), extent=[-O.XLimMeters, O.XLimMeters, 0, O.YLimMeters])
# plt.colorbar(img, ax=ax)
delta_psi = -2*pi/180
O.rotate_grid(delta_psi)
O2.rotate_grid_old(delta_psi)

ax2 = plt.subplot(132)
ax2.set(xlabel='X [m]', ylabel='Y [m])')
img = ax2.imshow(O.get_p(), extent=[-O.XLimMeters, O.XLimMeters, 0, O.YLimMeters])
# plt.colorbar(img, ax=ax2)

ax3 = plt.subplot(133)
ax3.set(xlabel='X [m]', ylabel='Y [m])')
img = ax3.imshow(O2.get_p(), extent=[-O.XLimMeters, O.XLimMeters, 0, O.YLimMeters])
# plt.colorbar(img, ax=ax3)
plt.show()