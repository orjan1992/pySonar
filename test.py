from readLogFile.readCsvFile import ReadCsvFile
log = '/home/orjangr/Repos/pySonar/logs/UdpHubLog_4001_2017_11_02_09_00_03.csv'
file = ReadCsvFile(log, 4002, 13102)

b = file.readNextMsg()
for i in range(0, 2000):
    msg = file.readNextMsg()
    if msg != -1 and msg != 0:
        print(msg.sensorStr)
    elif msg == -1:
        break


file.close()
