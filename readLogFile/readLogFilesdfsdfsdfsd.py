from math import pi
from struct import unpack

import numpy as np

from messages.sonarMsg import SonarMsg
from readLogFile.helperFunctions import Wrap2pi, getTimeLog


class ReadLogFile(object):
    """
    Reading sonar logfiles
    """

    header = ""
    version = 0
    regOffset = 0
    dataOffset = 0
    nScanLines = 0
    configOffset = 0
    extraOffset = 0
    indexOffset = 0
    checkOffset = 0
    GRAD2RAD = pi / (16 * 200)
    messagesReturned = 0

    def __init__(self, filename):
        self.binary_file = open(filename, "rb")
        data = self.binary_file.read(80)
        # Log header information
        (self.header, self.version, self.regOffset, self.dataOffset, self.nScanLines, self.configOffset,
         self.extraOffset, self.indexOffset, self.checkOffset, self.tOpen, self.tClose) = unpack('<32sIIIIIIIIdd', data)

        # scanLines
        self.binary_file.seek(self.dataOffset)

    def read_next_msg(self):
        if self.binary_file.tell() >= self.configOffset:
            print('End of scan lines reached!')
            return -1
        msg = SonarMsg('')
        data = self.binary_file.read(46)
        if len(data) < 46:
            return -1
        (msg.length, time, msg.txNode, msg.rxNode,
         msg.type, dummy1, dummy2, dummy3,
         dummy4, msg.headStatus, msg.sweepCode, msg.hdCtrl,
         msg.rangeScale, dummy5, msg.gain, msg.slope,
         msg.adSpan, msg.adLow, msg.headingOffset, msg.adInterval,
         msg.leftLim, msg.rightLim, msg.step, msg.bearing,
         msg.dataBins) = unpack('<HdBBBBBHBBBHHLBHBBHHHHBHH', data)
        # redefining vessel x as 0 deg and vessel starboard as +90
        msg.rightLim = Wrap2pi((msg.rightLim * self.GRAD2RAD + pi))
        msg.leftLim = Wrap2pi((msg.leftLim * self.GRAD2RAD + pi))
        msg.bearing = Wrap2pi((msg.bearing * self.GRAD2RAD + pi))
        msg.step = msg.step * self.GRAD2RAD
        msg.rangeScale = msg.rangeScale*0.1

        msg.date, msg.time, msg.date_time = getTimeLog(time)

        if (msg.type == 2) and (msg.bearing <= pi/2) and (msg.bearing >= -pi/2):
            data = self.binary_file.read(msg.dataBins)
            if msg.hdCtrl & 1:
                # adc8On bit is set
                msg.data = np.array(unpack(('<%iB' % msg.dataBins), data), dtype=np.uint8)
            else:
                tmp = unpack(('<%iB' % msg.dataBins), data)
                msg.data = np.zeros((len(tmp)*2, 1), dtype=np.uint8)
                for i in range(0, len(tmp)):
                    msg.data[2 * i] = (msg.data[i] & 240) >> 4 # 4 first bytes
                    msg.data[2 * i + 1] = msg.data[i] & 15 # 4 last bytes
            if (msg.headStatus >> 7) & 1: # bit 7 i set = extra appends message
                self.binary_file.seek(msg.length - 46 - msg.dataBins, 1)
            self.messagesReturned += 1
            return msg
        else:
            self.binary_file.seek(msg.length-46, 1)
            return 0

    def close(self):
        self.binary_file.close()
