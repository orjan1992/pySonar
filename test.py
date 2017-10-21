from readLogFile.readLogFile import ReadLogFile
a = ReadLogFile("logs/90degree scan target moving in.V4LOG")
#a = ReadLogFile("logs/360 degree scan harbour piles.V4LOG")
#print(a.scanLines[0]['data'])
from sys import getsizeof
getsizeof(a)
