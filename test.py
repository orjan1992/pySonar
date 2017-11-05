from readLogFile.readCsvFile import ReadCsvFile
# log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
# file = ReadCsvFile(log, 4002, 13102)
#
# for i in range(0, 2000):
#     msg = file.readNextMsg()
#     if msg != -1 and msg != 0:
#         print(msg.sensorStr)
#     elif msg == -1:
#         break
# file.close()
import math
from readLogFile.oGrid import OGrid
import matplotlib.pyplot as plt
grid = OGrid(0.1, 10, 5, 0.5)
step = 32*grid.GRAD2RAD
cells = grid.sonarConeLookup(step, math.pi/4)
grid.updateCells(cells, 1000)
cells = grid.sonarConeLookup(step, -math.pi/4)
grid.updateCells(cells, -1000)
fig, ax = grid.showP()
fig.show()
# #plotting
# fig, ax = plt.subplots()
# ax.imshow(grid.oLog, extent=[0, 1, 0, 1])
# plt.show()
