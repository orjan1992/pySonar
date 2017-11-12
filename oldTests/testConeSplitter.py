import math

from ogrid.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.readLogFile import ReadLogFile

csv = 0
if csv:
    log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
    file = ReadCsvFile(log, 4002, 13102)
else:
    log = 'logs/360 degree scan harbour piles.V4LOG'
    file = ReadLogFile(log)

O = OGrid(0.1, 20, 15, 0.5)

O.coneSplit(O.sonarConeLookup(8*O.GRAD2RAD, -math.pi/3), 5)
O.show()

#
# def coneSplit(self, cone, rangeScale):
#     subRange = cone[self.rHigh.flat[cone] < rangeScale - self.deltaSurface]
#     onRange = cone[self.rLow.flat[cone] < (rangeScale + self.deltaSurface)]
#     onRange = onRange[self.rHigh.flat[onRange] > (rangeScale - self.deltaSurface)]
#     above = cone[self.rLow.flat[cone] >= (rangeScale + self.deltaSurface)]
#     self.updateCells(subRange, 0.5)
#     self.updateCells(onRange, 2)
#     self.updateCells(above, 1)
#     # print(subRange)
#     # print(onRange)
#     # print(above)
#     print(np.intersect1d(subRange, onRange))
#     print(np.intersect1d(onRange, above))
#     print(np.intersect1d(subRange, above))