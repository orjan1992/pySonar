import csv
from itertools import islice
from readLogFile.posMsg import PosMsg
import binascii

class ReadCsvFile(object):
    """
    Read CSV log files
    """

    nRows = 0
    lastMsg = 0

    def __init__(self, filename, sonarPort, posPort):
        self.file = open(filename, newline='')
        self.reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])

        self.sonarPort = sonarPort
        self.posPort = posPort


    def close(self):
        self.file.close()

    def readRows(self, start, n = 1):
        """
        Read the next rows
        :param start: rows
        :param n: number of rows to read, default=1
        :return: ordered dict of rows. fields: time, ip, port, data
        """
        # self.reader.seek(0)
        tmp = list(islice(self.reader, start, start + n))
        self.lastMsg = start+n
        return tmp


    def readNextRow(self):
        """
        Read next position msg
        :return: msg
        """
        self.lastMsg = self.lastMsg+1
        return list(islice(self.reader, 0, 1))

    def readNextMsg(self):
        """
        Read next msg
        :return: msg
        """
        msg = self.readRows(self.lastMsg, 1)
        if int(msg[0]['port']) == self.sonarPort:
            sonar = True
            i = 0
            while sonar and i<12:
                tmp = self.readNextRow()
                if int(tmp[0]['port']) == self.sonarPort:
                    msg.append(tmp)
                    i = i+1
                else:
                    sonar = False
                    self.lastMsg = self.lastMsg - 1
            return self.parseSonarMsg(msg)
        elif int(msg[0]['port']) == self.posPort:
            return self.parsePosMsg(msg)
        else:
            return -1

    def parsePosMsg(self, raw_msg):
        msg = PosMsg(raw_msg[0]['time'])
        data = str(binascii.unhexlify(''.join(raw_msg[0]['data'].split()))).split(',')
        msg.id = data[0]
        msg.head = data[1]
        msg.roll = data[2]
        msg.pitch = data[3]
        msg.depth = data[4]
        msg.alt = data[5]
        msg.lat = data[6]
        msg.long = data[7]
        return msg


    def parseSonarMsg(self, msg):
        for row in msg:
            print(row, '\n')