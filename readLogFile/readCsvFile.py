import csv
from itertools import islice
from readLogFile.posMsg import PosMsg
from readLogFile.sonarMsg import SonarMsg
import binascii
import struct

class ReadCsvFile(object):
    """
    Read CSV log files
    """
    SONAR_START = 64 # 0x40 = 64 = '@'
    LINE_FEED = 10 # 0x0A = 10 = 'LF'
    curSonarMsg = bytearray()

    def __init__(self, filename, sonarPort, posPort):
        self.file = open(filename, newline='')
        self.reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])

        self.sonarPort = sonarPort
        self.posPort = posPort


    def close(self):
        self.file.close()

    def readNextRow(self):
        """
        Read next msg
        :return: msg
        """
        return list(islice(self.reader, 0, 1))

    def readNextMsg(self):
        """
        Read next msg
        :return: msg
        """
        msg = self.readNextRow()
        try:
            if int(msg[0]['port']) == self.sonarPort:
                sonar = True
                # i = 0
                # while sonar and i<12:
                #     tmp = self.readNextRow()
                #     if int(tmp[0]['port']) == self.sonarPort:
                #         msg.append(tmp[0])
                #         i = i+1
                #     else:
                #         sonar = False
                return self.splitSonarMsg(msg)
            elif int(msg[0]['port']) == self.posPort:
                return self.parsePosMsg(msg)
            else:
                return -1
        except IndexError:
            print('End of file reached')
            return -1

    def parsePosMsg(self, raw_msg):
        msg = PosMsg(raw_msg[0]['time'])
        data = str(binascii.unhexlify(''.join(raw_msg[0]['data'].split()))).split(',')
        # print(data)
        msg.id = data[0]
        msg.head = data[1]
        msg.roll = data[2]
        msg.pitch = data[3]
        msg.depth = data[4]
        msg.alt = data[5]
        msg.lat = data[6]
        msg.long = data[7]
        return msg


    def splitSonarMsg(self, msg):
        bArray = bytearray.fromhex(''.join(l['data'] for l in msg))
        # print(bArray)
        # print(bArray[0])
        length = len(bArray)
        noEnd = True
        for i in range(0, length):
            if bArray[i] == self.LINE_FEED and i < length and bArray[i+1] == self.SONAR_START:
                self.curSonarMsg = b''.join([self.curSonarMsg, bArray[0:(i+1)]])
                self.parseSonarMsg()
                self.curSonarMsg = bArray[(i+1):length]
                print('new %s'%self.curSonarMsg)
                noEnd = False
                break
        if noEnd:
            self.curSonarMsg = b''.join([self.curSonarMsg, bArray])

            # if bArray[i] == self.SONAR_START:
            #     # print(bArray[(i+1):(i+5)])
            #     # for b in bArray[(i+1):(i+5)]:
            #     #     print(b)
            #     print('sum ' + str(sum(bArray[(i+1):(i+5)])))
            #     print(struct.unpack('H', bArray[(i+5):(i+7)]))
            #     print(bArray[i+7])
        # data = str(binascii.unhexlify(''.join(msg[0]['data'].split()))).split(',')
        # print(data)
        # if bArray[0] == 2:
        #     mtHeadData
        #     print('Data')
        # elif bArray[0] == 15:
        #     print('mtTimeout')
        # elif bArray[0] == 0:
        #     print('mtNull')
        # else:
        #     print('Unknown: ' +str(bArray[0]))

        # print(struct.unpack('B',byte(bArray[0]))[0])
        return 1
    #sum er 128 eller 200 dec fra hex
    #binary word 41 03 = 0x341= 833 dec
    def parseSonarMsg(self):
        if self.curSonarMsg[0] != self.SONAR_START:
            print('Message not complete')
            return -1
        else:


