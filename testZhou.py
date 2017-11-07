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
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = OGrid(0.1, 20, 15, 0.5)
Threshold = 60
theta = np.zeros(1)
while file.messagesReturned < 2000:
    msg = file.readNextMsg()
    if type(msg) is SonarMsg and msg.type == 2:
        O.autoUpdateZhou(msg, Threshold)
        theta = np.append(theta, msg.bearing)
    elif msg == -1:
        break
file.close()
#
print('Messages returned: %i\n' % file.messagesReturned)
fig, ax = O.showP()
plt.show()

# (fig, ax) = plt.subplots()
# ax.plot(range(0, np.shape(theta)[0]), theta)
# plt.show()
