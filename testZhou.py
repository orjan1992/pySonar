import math
from readLogFile.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.sonarMsg import SonarMsg
from readLogFile.posMsg import PosMsg
from readLogFile.sensor import Sensor
O = OGrid(0.1, 10, 10, 0.5)
log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
file = ReadCsvFile(log, 4002, 13102)
Threshold =60
for i in range(0, 2000):
    msg = file.readNextMsg()
    if msg != -1 and msg != 0 and msg.sensor == 2:
        # msg = SonarMsg(msg)
        dl = msg.rangeScale/len(msg.data)
        step = msg.step
        theta = msg.bearing
        nonUpdatedCells = O.sonarConeLookup(msg.step, theta)
        distanceUpdated = False
        for j in range(0, len(msg.data)):
            if abs((j * dl) * math.sin(theta)) > O.XLimMeters or abs((j * dl) * math.cos(theta)) > O.YLimMeters:
                break # SJEKK DETTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if msg.data[j] > Threshold:
                nonUpdatedCells = O.updateCellsZhou2(nonUpdatedCells, j * dl, theta)
                distanceUpdated = True
        if not distanceUpdated:
            O.updateCellsZhou2(nonUpdatedCells, math.inf, theta)
    elif msg == -1:
        break
file.close()

fig, ax = O.showP()
fig.show()