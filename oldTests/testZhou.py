import numpy as np

from messages.sonarMsg import SonarMsg
from ogrid.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.readLogFile import ReadLogFile

csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    log = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = OGrid(0.1, 20, 15, 0.5, True)
Threshold = 60
theta = np.zeros(1)
while file.messagesReturned < 2000:
    msg = file.read_next_msg()
    if type(msg) is SonarMsg and msg.type == 2:
        print(file.messagesReturned)
        O.auto_update_zhou(msg, Threshold)
        theta = np.append(theta, msg.bearing)
    elif msg == -1:
        break
    if file.messagesReturned % 5 == 0:
        O.showP()
file.close()
#
print('Messages returned: %i\n' % file.messagesReturned)