import binascii
import struct
import logging
import numpy as np
from math import pi, sin, cos, sqrt, atan2
import re
from settings import ConnectionSettings
from enum import Enum

from messages.sensor import Sensor
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
    psi = 0.0
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
            self.psi = float(str_array[1])*np.pi / 180.0
            self.roll = float(str_array[2])*np.pi / 180.0
            self.pitch = float(str_array[3])*np.pi / 180.0
            # self.depth = float(str_array[4])
            # self.alt = float(str_array[5])
            # self.lat = float(str_array[6])
            # self.long = float(str_array[7])
            self.alt = float(str_array[4])
            self.lat = float(str_array[5])
            self.long = float(str_array[6])
        except:
            logger.info('NMEA msg to short')
            self.error = True
            return

    def __sub__(self, other):
        lat_diff = self.lat - other.lat
        # print('(self {}) - (other {}) = {}'.format(self.lat, other.lat, lat_diff))
        long_diff = self.long - other.long
        alpha = atan2(long_diff, lat_diff)
        dist = sqrt(lat_diff ** 2 + long_diff ** 2)
        dpsi = self.psi - other.psi

        dx = cos(alpha - self.psi) * dist
        dy = sin(alpha - self.psi) * dist
        return UdpPosMsgDiff(dx, dy, dpsi)

    def __str__(self):
        return 'psi: {}, roll: {}, pitch: {}, alt: {}, lat: {}, long: {}'.format(self.psi, self.roll, self.pitch,
                                                                                   self.alt, self.lat, self.long)

class UdpPosMsgDiff:
    def __init__(self, dx, dy, dpsi):
        self.dx = dx
        self.dy = dy
        self.dpsi = dpsi

    def __add__(self, other):
        _dx = self.dx + other.dx
        _dy = self.dy + other.dy
        _dpsi = self.dpsi + other.dpsi
        logger.debug('self={}, other={}, sum={}'.format(self.dx, other.dx, _dx))
        return UdpPosMsgDiff(_dx, _dy, _dpsi)

    def __str__(self):
        return 'dx: {},\tdy: {}\t, dpsi: {}'.format(self.dx, self.dy, self.dpsi * 180 / pi)


class AutoPilotBinary:
    sid = 0
    msg_id = 0
    payload = None

    def compile(self):
        if self.payload is None:
            return struct.pack('ihh', len(self.payload), self.sid, self.msg_id)
        else:
            msg = bytearray(8+len(self.payload))
            msg[:8] = struct.pack('ihh', len(self.payload), self.sid, self.msg_id)
            msg[8:] = self.payload
            return msg

    @staticmethod
    def parse(msg):
        try:
            length, sid, msg_id = struct.unpack('ihh', msg[:8])
        except struct.error:
            raise CorruptMsgException
        except IndexError:
            raise CorruptMsgException
        if msg_id == 18:
            try:
                return AutoPilotRemoteControlRequestReply(msg[8:], length, sid, msg_id)
            except IndexError:
                raise CorruptMsgException
        else:
            raise OtherMsgTypeException


class AutoPilotCommand(AutoPilotBinary):
    msg_id = 2

    def __init__(self, option, sid=0):
        self.sid = sid
        self.payload = struct.pack('i', option.value)

class AutoPilotCommandOptions(Enum):
    NONE = 0
    START = 1
    STOP = 2
    SAVE = 3
    CLEAR_WPS = 6

class AutoPilotTrackingSpeed(AutoPilotBinary):
    msg_id = 3

    def __init__(self, speed, sid=0):
        self.sid = sid
        self.payload = struct.pack('f', speed)

class AutoPilotAddWaypoints(AutoPilotBinary):
    msg_id = 4

    def __init__(self, wp_list, sid=0):
        """

        :param wp_list: list of (N, E, D)
        :param sid:
        """
        self.sid = sid
        self.payload = bytearray(4 * (1 + 3 * len(wp_list)))
        self.payload[:4] = struct.pack('i', len(wp_list))
        for i in range(len(wp_list)):
            self.payload[4 + i*12:4 + (i+1)*12] = struct.pack('fff', wp_list[i][0], wp_list[i][1], wp_list[i][2])


class AutoPilotTrackingConfig(AutoPilotBinary):
    msg_id = 11

    def __init__(self, roa, braking_zone, gain_speed, lookahead, gain_path, cornering_speed, sid=0):
        self.sid = sid
        self.payload = struct.pack('ffffff', roa, braking_zone, gain_path, lookahead, gain_path, cornering_speed)

class AutoPilotGuidanceMode(AutoPilotBinary):
    msg_id = 14

    def __init__(self, mode, sid=0):
        self.sid = sid
        self.payload = struct.pack('i', mode.value)

class AutoPilotGuidanceModeOptions(Enum):
    NONE = 0
    STATION_KEEPING = 1
    CIRCULAR_INSPECTION_MODE = 3
    PATH_FOLLOWING = 5

class AutoPilotRemoteControlRequest(AutoPilotBinary):
    msg_id = 17

    def __init__(self, sid):
        self.sid = sid

class AutoPilotRemoteControlRequestReply(AutoPilotBinary):
    msg_id = 18
    acquired = False

    def __init__(self, msg, length, sid, msg_id):
        if len(msg) != length or msg_id != self.msg_id:
            raise CorruptMsgException
        try:
            self.token, status = struct.unpack('hB', msg)
            if self.token > 0 and status == 1:
                self.acquired = True
        except struct.error:
            raise CorruptMsgException

class AutoPilotGetMessage(AutoPilotBinary):
    msg_id = 1000

    def __init__(self, msg_ids, sid=0):
        self.sid = sid
        if msg_ids is list:
            self.payload = struct.pack('{}h'.format(len(msg_ids)), *msg_ids)
        else:
            self.payload = struct.pack('h', msg_ids)

class UncompleteMsgException(Exception):
    pass


class CorruptMsgException(Exception):
    pass


class OtherMsgTypeException(Exception):
    pass


if __name__ == '__main__':
    wp_list = [[5.0, 20.0, 12], [2.0, 2.0, 1.0]]
    msg = AutoPilot_AddWaypoints(wp_list, 10)
    a = msg.compile()
    b = struct.unpack('IHHIffffff', a)
    print(b)