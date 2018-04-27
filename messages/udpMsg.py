import binascii
import struct
import logging
import numpy as np
from math import pi, sin, cos, sqrt, atan2
import re
from settings import ConnectionSettings, Settings
from enum import Enum

from messages.sensor import Sensor
from settings import Map
logger = logging.getLogger('SonarMsg')


class MtHeadData(Sensor):
    sensorStr = 'SONAR'
    sensor = 2
    GRAD2RAD = pi / (16 * 200)

    def __init__(self, byte_array, *kwargs):
        super().__init__(*kwargs)
        try:
            if byte_array[0] != 0x40:
                raise CorruptMsgException('Wrong start byte')
            try:
                hex_length = b''.join([binascii.unhexlify(byte_array[3:5]), binascii.unhexlify(byte_array[1:3])])
            except:
                raise  CorruptMsgException()
            hex_length = struct.unpack('H', hex_length)
            word_length = struct.unpack('H', byte_array[5:7])
            if hex_length != word_length:
                raise CorruptMsgException('from hex_length != word_length')

            self.sid = byte_array[7]
            self.did = byte_array[8]
            self.count = byte_array[9]
            self.msg_type = byte_array[10]
            self.seq_bitset = byte_array[11]
            if self.seq_bitset != 0x80:  # Message Sequence Bitset (see below).
                logger.error('Multipacket message. Not implemented {:b}'.format(byte_array[11]))
                NotImplementedError('Multipacket message. Not implemented {:b}'.format(byte_array[11]))
            if self.msg_type != 0x02:
                raise OtherMsgTypeException
            # mtHeadData
            if byte_array[12] != self.sid:
                logger.info('Sonar msg error: Tx1 != Tx2')
                raise CorruptMsgException

            (self.total_count, self.device_type, self.head_status,
             self.sweep_code, self.hd_ctrl, self.range_scale) = struct.unpack('<HBBBHH', byte_array[13:22])

            self.adc8on = (self.hd_ctrl & 1) == 1
            self.cont = (self.hd_ctrl & (1 << 1)) != 0
            self.scan_right = (self.hd_ctrl & (1 << 2)) != 0
            self.invert = (self.hd_ctrl & (1 << 3)) != 0
            # self.mot_off = (self.hd_ctrl & (1 << 4)) != 0
            # self.tx_off = (self.hd_ctrl & (1 << 5)) != 0
            # Spare bit
            self.chan2 = (self.hd_ctrl & (1 << 7)) != 0
            # self.raw = (self.hd_ctrl & (1 << 8)) != 0
            # self.has_mot = (self.hd_ctrl & (1 << 9)) != 0
            # self.apply_offset = (self.hd_ctrl & (1 << 10)) != 0
            self.ping_pong = (self.hd_ctrl & (1 << 11)) != 0
            self.stare_l_lim = (self.hd_ctrl & (1 << 12)) != 0
            # self.reply_asl = (self.hd_ctrl & (1 << 13)) != 0
            # self.reply_thr = (self.hd_ctrl & (1 << 14)) != 0
            # self.ignore_sensor = (self.hd_ctrl & (1 << 15)) != 0

            # 7 bytes, TxN = 4byte, gain = 1byte, slope = 2byte
            (self.ad_span, self.ad_low) = struct.unpack('<BB', byte_array[29:31])
            # 2bytes, heading offset
            (self.ad_interval, self.l_lim, self.r_lim, self.step,
             self.bearing, self.dbytes) = struct.unpack('<HHHBHH', byte_array[33:44])

            if self.adc8on:
                if len(byte_array) < 44+self.dbytes:
                    raise UncompleteMsgException("To few databytes")
                self.data = np.array(list(byte_array[44:(44+self.dbytes)]), dtype=np.uint8)
            else:
                if len(byte_array) < 44+self.dbytes:
                    raise UncompleteMsgException
                tmp = struct.unpack(('<%iB' % self.dbytes), byte_array[44:(44 + self.dbytes)])
                self.data = np.zeros((len(tmp) * 2, 1), dtype=np.uint8)
                for i in range(0, len(tmp)):
                    self.data[2 * i] = (self.data[i] & 240) >> 4  # 4 first bytes
                    self.data[2 * i + 1] = self.data[i] & 15  # 4 last bytes

            # Convert to similar format as moosmsg
            self.range_scale *= 0.1
            self.length = self.dbytes
            try:
                self.extra_bytes = byte_array[43+self.dbytes:]
            except:
                self.extra_bytes = None

            if Settings.inverted_sonar:
                self.bearing = 6400 - self.bearing
            # self.time = 0
            # if byte_array[43 + self.dbytes] != 10:
            #     logger.error('No end of message')
            #     raise CorruptMsgException
        except IndexError:
            raise UncompleteMsgException("Index error")
        except struct.error:
            raise UncompleteMsgException("Struct error")
        except binascii.Error:
            raise UncompleteMsgException("binascii error")
        # except Exception as e:
        #     print(e)
        #     raise CorruptMsgException

class UdpPosMsg(Sensor):

    sensor = 1
    sensorStr = 'Position'
    id = 0
    yaw = 0.0
    roll = 0.0
    pitch = 0.0
    depth = 0.0
    lat = 0.0
    long = 0.0

    error = False

    def __init__(self, data):
        string = data.decode('ascii')
        str_array = re.split('\*|,', string)
        if str_array[0] != '$ROV':
            logger.info('Not NMEA msg from ROV')
            self.error = True
            return
        # NMEA Checksum
        if ConnectionSettings.use_nmea_checksum:
            i = 1
            checksum = 0
            while i < len(data) and data[i] != 0x2a:
                checksum ^= data[i]
                i += 1
            if int(str_array[-1], 16) != checksum:
                logger.info('Invalid checksum')
                self.error = True
                return
        try:
            self.yaw = float(str_array[1])*np.pi / 180.0
            self.roll = float(str_array[2])*np.pi / 180.0
            self.pitch = float(str_array[3])*np.pi / 180.0
            # self.depth = float(str_array[4])
            # self.alt = float(str_array[5])
            # self.lat = float(str_array[6])
            # self.long = float(str_array[7])
            self.alt = float(str_array[4])
            self.lat = float(str_array[5])
            self.long = float(str_array[6])
            self.north, self.east = Map.map_proj(self.lat, self.long)
        except:
            logger.info('NMEA msg to short')
            self.error = True
            return

    def __sub__(self, other):
        north_diff = self.north - other.north
        # print('(self {}) - (other {}) = {}'.format(self.north, other.north, north_diff))
        east_diff = self.east - other.east
        alpha = atan2(east_diff, north_diff)
        dist = sqrt(north_diff ** 2 + east_diff ** 2)
        dyaw = self.yaw - other.yaw

        dx = cos(alpha - self.yaw) * dist
        dy = sin(alpha - self.yaw) * dist
        return UdpPosMsgDiff(dx, dy, dyaw)

    def __str__(self):
        return 'yaw: {}, roll: {}, pitch: {}, alt: {}, north: {}, east: {}'.format(self.yaw, self.roll, self.pitch,
                                                                                   self.alt, self.north, self.east)

    def to_tuple(self):
        return self.north, self.east, self.alt, self.yaw

class UdpPosMsgDiff:
    def __init__(self, dx, dy, dyaw):
        self.dx = dx
        self.dy = dy
        self.dyaw = dyaw

    def __add__(self, other):
        _dx = self.dx + other.dx
        _dy = self.dy + other.dy
        _dyaw = self.dyaw + other.dyaw
        logger.debug('self={}, other={}, sum={}'.format(self.dx, other.dx, _dx))
        return UdpPosMsgDiff(_dx, _dy, _dyaw)

    def __str__(self):
        return 'dx: {},\tdy: {}\t, dyaw: {}'.format(self.dx, self.dy, self.dyaw * 180 / pi)


class UncompleteMsgException(Exception):
    pass


class CorruptMsgException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OtherMsgTypeException(Exception):
    pass


if __name__ == '__main__':
    wp_list = [[5.0, 20.0, 12], [2.0, 2.0, 1.0]]
    msg = AutoPilot_AddWaypoints(wp_list, 10)
    a = msg.compile()
    b = struct.unpack('IHHIffffff', a)
    print(b)