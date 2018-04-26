import matplotlib.pyplot as plt
import numpy as np

from messages.udpMsg import SonarMsg
from ogrid.rawGrid import RawGrid
from oldTests.readLogFile import ReadCsvFile
from oldTests.readLogFile import ReadLogFile

csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = RawGrid(0.1, 20, 15, 0.5)
Threshold = 60
theta = np.array(1)
time = np.array(4.244359902158565*10**4)
fig, ax = plt.subplots()

while file.messagesReturned < 200:
    msg = file.read_next_msg()
    if type(msg) is SonarMsg:

        theta = np.append(theta, msg.bearing)
        time = np.append(time, msg.time)
        ax.plot(range(0, np.shape(msg.data)[0]), msg.data)
    elif msg == -1:
        break
file.close()


(fig, ax) = plt.subplots()
ax.plot(range(0, np.shape(theta)[0]), theta)
plt.show()
