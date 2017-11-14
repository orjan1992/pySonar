import binascii
import csv
import struct
from itertools import islice
from math import pi
import numpy as np
from datetime import datetime, timedelta
from os import walk


from messages.sonarMsg import SonarMsg
from messages.posMsg import PosMsg
from readLogFile.helperFunctions import Wrap2pi, getTimeCsv, get_time_csv_file_name
import logging; logger = logging.getLogger('readLogFile.ReadCsvFile')


class ReadCsvFile(object):
    """
    Read CSV log files
    """
    SONAR_START = 64 # 0x40 = 64 = '@'
    LINE_FEED = 10 # 0x0A = 10 = 'LF'
    GRAD2RAD = pi / (16 * 200)
    curSonarMsg = bytearray()
    curSonarMsgTime = ''
    messagesReturned = 0

    def __init__(self, filename, sonarPort =4002, posPort=13102, cont_reading=False):
        self.file = open(filename, newline='')
        self.reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])
        self.cont_reading = cont_reading
        if cont_reading:
            tmp = filename.split('/')
            self.file_path = '/'.join(tmp[:-1])
            cur_file = tmp[-1]
            self.file_list = []
            for (dirpath, dirnames, filenames) in walk(self.file_path):
                self.file_list.extend(filenames)
                break
            self.file_list.sort()
            self.file_number = self.file_list.index(cur_file)
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

    def read_next_msg(self):
        """
        Read next msg
        :return: msg
        """
        msg = self.readNextRow()
        try:
            if int(msg[0]['port']) == self.sonarPort:
                return self.splitSonarMsg(msg)
            elif int(msg[0]['port']) == self.posPort:
                return self.parsePosMsg(msg)
            else:
                logger.info('Message from other source read')
                return 0
        except IndexError:
            logger.info('End of file reached')
            if self.cont_reading:
                return self.start_again()
            else:
                return -1

    def start_again(self):
        try:
            self.file_number += 1
            self.file = open('/'.join([self.file_path, self.file_list[self.file_number]]), newline='')
            self.reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])
            logger.info('Started on file: {}'.format(self.file_list[self.file_number]))
            return self.read_next_msg()
        except FileNotFoundError:
            logger.info('No more files to open')
            return -1


    def parsePosMsg(self, raw_msg):
        msg = PosMsg(getTimeCsv(raw_msg[0]['time']))
        data = str(binascii.unhexlify(''.join(raw_msg[0]['data'].split()))).split(',')
        # print(data)
        msg.id = data[0]
        msg.head = float(data[1])*pi/180
        msg.roll = float(data[2])*pi/180
        msg.pitch = float(data[3])*pi/180
        msg.depth = float(data[4])
        msg.alt = float(data[5])
        msg.lat = float(data[6])
        msg.long = float(data[7])
        return msg


    def splitSonarMsg(self, msg):
        bArray = bytearray.fromhex(''.join(l['data'] for l in msg))
        # print(bArray)
        # print(bArray[0])
        length = len(bArray)
        if bArray[0] != self.SONAR_START:
            for i in range(0, length):
                if bArray[i] == self.LINE_FEED and i < length and bArray[i+1] == self.SONAR_START:
                    self.curSonarMsg = b''.join([self.curSonarMsg, bArray[:(i+1)]])
                    returnMsg = self.parseSonarMsg()
                    self.curSonarMsg = bArray[(i+1):length]
                    self.curSonarMsgTime = msg[0]['time']
                    return returnMsg
                if bArray[i] == self.LINE_FEED and i == length:
                    self.curSonarMsg = b''.join([self.curSonarMsg, bArray])
                    returnMsg = self.parseSonarMsg()
                    # TODO not correct time
                    self.curSonarMsgTime = msg[0]['time']
                    self.curSonarMsg = b''
                    return returnMsg
        self.curSonarMsg = b''.join([self.curSonarMsg, bArray])
        return 0

    def parseSonarMsg(self):
        if self.curSonarMsg[0] != self.SONAR_START:
            logger.error('First byte of sonar msg is not correct. Logfile probably started in the middle of a message')
            return 0
        else:
            hexLength = b''.join([binascii.unhexlify(self.curSonarMsg[3:5]), binascii.unhexlify(self.curSonarMsg[1:3])])
            hexLength = struct.unpack('H', hexLength)
            wordLength = struct.unpack('H', self.curSonarMsg[5:7])
            if hexLength != wordLength:
                logger.error('Sonar msg error: hex {} \t word {}\t len(bytestring) {}'.format(hexLength, wordLength, len(self.curSonarMsg)))
                # should return some error
                return 0
            if (hexLength[0] + 6) < len(self.curSonarMsg):
                logger.info('Sonar msg error, probably started new file with'
                            ' incomplete message: hex {} \t word {}\t len(bytestring) {}'.format(hexLength, wordLength,
                                                                                              len(self.curSonarMsg)))
                return 0
            msg = SonarMsg(getTimeCsv(self.curSonarMsgTime))
            msg.txNode = self.curSonarMsg[7]
            msg.rxNode = self.curSonarMsg[8]
            # self.curSonarMsg[9] Byte Count of attached message that follows this byte.
            # Set to 0 (zero) in ‘mtHeadData’ reply to indicate Multi-packet mode NOT used by device.
            msg.type = self.curSonarMsg[10]
            if self.curSonarMsg[11] != 128:  #Message Sequence Bitset (see below).
                logger.error('Multipacket message. Not implemented {:b}'.format(self.curSonarMsg[11]))
                NotImplementedError('Multipacket message. Not implemented {:b}'.format(self.curSonarMsg[11]))
            if msg.type == 2:
                # mtHeadData
                if self.curSonarMsg[12] != msg.txNode:
                    logger.info('Sonar msg error: Tx1 != Tx2')
                    return 0
                # 13-14 Total Byte Count of Device Parameters + Reply Data (all packets).
                # msg.deviceType = self.curSonarMsg[15]
                # msg.headStaus = self.curSonarMsg[16]
                # msg.sweepCode = self.curSonarMsg[17]
                # msg.hdCtrl = self.curSonarMsg[18:20]
                # msg.rangeScale = struct.unpack('H', self.curSonarMsg[20:22])
                (msg.deviceType, msg.headStatus,
                 msg.sweepCode, msg.hdCtrl,
                 msg.rangeScale, dummy,
                 msg.gain, msg.slope,
                 msg.adSpan, msg.adLow,
                 msg.headingOffset, msg.adInterval,
                 msg.leftLim, msg.rightLim,
                 msg.step, msg.bearing,
                 msg.dataBins) = struct.unpack('<BBBHHIBHBBHHHHBHH', self.curSonarMsg[15:44])
                if not(msg.deviceType in [2, 3, 5, 11, 17]):
                    logger.info('Msg contained wrong device type. Type is: {}'.format(msg.deviceType))
                    return 0
                # redefining vessel x as 0 deg and vessel starboard as +90
                msg.rightLim = Wrap2pi((msg.rightLim*self.GRAD2RAD+pi))
                msg.leftLim = Wrap2pi((msg.leftLim*self.GRAD2RAD+pi))
                msg.bearing = Wrap2pi((msg.bearing*self.GRAD2RAD+pi))
                msg.step = msg.step * self.GRAD2RAD
                msg.rangeScale = msg.rangeScale*0.1
                if msg.hdCtrl & 1:
                    #adc8On bit is set
                    msg.data = np.array(list(self.curSonarMsg[44:(hexLength[0]+5)]), dtype=np.uint8)
                else:
                    tmp = struct.unpack(('<%iB' % msg.dataBins), self.curSonarMsg[44:(hexLength[0]+5)])
                    msg.data = np.zeros((len(tmp) * 2, 1), dtype=np.uint8)
                    for i in range(0, len(tmp)):
                        msg.data[2 * i] = (msg.data[i] & 240) >> 4  # 4 first bytes
                        msg.data[2 * i + 1] = msg.data[i] & 15  # 4 last bytes
                if self.curSonarMsg[hexLength[0]+5] != 10:
                    logger.error('No end of message')
                    return 0
            else:
                logger.error('Other Sonar messagetypes not implemented. Msg type: %i' % msg.type)
                raise NotImplementedError('Other Sonar messagetypes not implemented. Msg type: %i' % msg.type)
            self.messagesReturned += 1
            return msg


