import math
from readLogFile.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.sonarMsg import SonarMsg
from readLogFile.readLogFile import ReadLogFile
import matplotlib.pyplot as plt
import numpy as np
csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)
    print(file.dataOffset)

O = OGrid(0.1, 20, 15, 0.5)
Threshold = 60
theta = np.array(1)
time = np.array(4.244359902158565*10**4)
for i in range(0, 60000):
    msg = file.readNextMsg()
    if type(msg) is SonarMsg:

        theta = np.append(theta, msg.bearing)
        time = np.append(time, msg.time)
    elif msg == -1:
        break
file.close()


(fig, ax) = plt.subplots()
ax.plot(range(0, np.shape(time)[0]), time)
fig.show()
print(time)