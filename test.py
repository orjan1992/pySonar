from readLogFile.readCsvFile import ReadCsvFile
log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
file = ReadCsvFile(log)
print(str(file.nRows))
file.readRows(10, )

file.close()