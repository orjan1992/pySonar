import time

import numpy as np

from messages.sonarMsg import SonarMsg
from ogrid.rawGrid import RawGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.readLogFilesdfsdfsdfsd import ReadLogFile

csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    log = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = RawGrid(0.1, 20, 15, 0.5)
O2 = RawGrid(0.1, 20, 15, 0.5)
Threshold = 60
theta = np.zeros(1)
while file.messagesReturned < 50:
    msg = file.read_next_msg()
    if type(msg) is SonarMsg and msg.type == 2:
        O.auto_update_zhou(msg, Threshold)
        O2.auto_update_zhou(msg, Threshold)
        theta = np.append(theta, msg.bearing)
    elif msg == -1:
        break
print('All equal before {}'.format(np.all(O.grid == O2.grid)))

t0 = time.time()
for i in range(0, 200):
    O.translational_motion(0.01, -0.01)
print('Fast time {:.2f}'.format(time.time()-t0))
t0 = time.time()
for i in range(0, 200):
    O2.trans2(0.01, -0.01)
print('Slow time {:.2f}'.format(time.time()-t0))
test = O.grid - O2.grid
print('All equal after {}'.format(np.all(O.grid == O2.grid)))