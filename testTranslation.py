import math
from readLogFile.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.sonarMsg import SonarMsg
from readLogFile.readLogFile import ReadLogFile
from matplotlib import pyplot as plt
import time
import numpy as np
csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    log = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = OGrid(0.1, 20, 15, 0.5)
Threshold = 60
theta = np.zeros(1)
while file.messagesReturned < 2:
    msg = file.readNextMsg()
    if type(msg) is SonarMsg and msg.type == 2:
        print(file.messagesReturned)
        O.autoUpdateZhou(msg, Threshold)
        theta = np.append(theta, msg.bearing)
    elif msg == -1:
        break
file.close()
ax = plt.subplot(211)
ax.set(xlabel='X [m]', ylabel='Y [m])')
img = ax.imshow(O.getP(), extent=[-O.XLimMeters, O.XLimMeters, 0, O.YLimMeters])
plt.colorbar(img, ax=ax)

O.translational_motion(-5, -5)


ax2 = plt.subplot(212)
ax2.set(xlabel='X [m]', ylabel='Y [m])')
img = ax2.imshow(O.getP(), extent=[-O.XLimMeters, O.XLimMeters, 0, O.YLimMeters])
plt.colorbar(img, ax=ax2)
plt.show()