import binascii
import struct
import logging
import numpy as np
from math import pi

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
                raise UncompleteMsgException()
            hex_length = b''.join([binascii.unhexlify(byte_array[3:5]), binascii.unhexlify(byte_array[1:3])])
            hex_length = struct.unpack('H', hex_length)
            word_length = struct.unpack('H', byte_array[5:7])
            if hex_length != word_length:
                raise CorruptMsgException

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
            try:
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
                    self.data = np.array(list(byte_array[44:(44+self.dbytes)]), dtype=np.uint8)
                else:
                    tmp = struct.unpack(('<%iB' % self.dbytes), byte_array[44:(hex_length[0] + 5)])
                    self.data = np.zeros((len(tmp) * 2, 1), dtype=np.uint8)
                    for i in range(0, len(tmp)):
                        self.data[2 * i] = (self.data[i] & 240) >> 4  # 4 first bytes
                        self.data[2 * i + 1] = self.data[i] & 15  # 4 last bytes
            except UncompleteMsgException:
                raise UncompleteMsgException
            # if byte_array[44 + self.dbytes] != 10:
            #     logger.error('No end of message')
            #     raise CorruptMsgException

            # redefining vessel x as 0 deg and vessel starboard as +90
            self.right_lim_rad = wrap2pi((self.r_lim * self.GRAD2RAD + pi))
            self.left_lim_rad = wrap2pi((self.l_lim * self.GRAD2RAD + pi))
            self.bearing_rad = wrap2pi((self.bearing * self.GRAD2RAD + pi))
            self.step_rad = self.step * self.GRAD2RAD
            self.range_scale_m = self.range_scale * 0.1
        except OtherMsgTypeException:
            raise OtherMsgTypeException
        except Exception:
            raise CorruptMsgException


def wrap2pi(angle):
    return (angle + pi) % (2 * pi) - pi


class UncompleteMsgException(Exception):
    pass


class CorruptMsgException(Exception):
    pass


class OtherMsgTypeException(Exception):
    pass
