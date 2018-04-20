from enum import Enum
import struct
from messages.udpMsg import CorruptMsgException, OtherMsgTypeException, UdpPosMsg

class MsgType(Enum):
    ERROR = 0
    WARNING = 1
    COMMAND_MESSAGE = 2
    CRUISE_SPEED = 3
    TRACKING_SPEED = 4
    ADD_WAYPOINTS = 5
    SETPOINT = 6
    CONTROLLER_TUNING = 7
    SET_GUIDANCE_MODE = 8
    SETPOINT_VERTICAL = 9
    WARNING_GUIDANCE = 10
    REMOTE_CONTROL_REQUEST = 11
    REMOTE_CONTROL_REQUEST_REPLY = 12
    EMERGENCY_BREAK = 13
    EMPTY_MESSAGE = 14
    CONTROL_VECTOR = 15
    GET_MESSAGE = 1000
    AUTOPILOT_STATUS = 1001
    ROV_STATE = 1002
    ROV_STATE_DESIRED = 1003
    INS_STATUS = 1004
    VERSION = 1005
    CONTROLLER_OPTIONS = 1006
    CONTROLLER_CONFIGURATION = 1007
    POSITION_REFERENCE_CONFIGURATION = 1008
    VELOCITY_REFERENCE_CONFIGURATION = 1009
    PATH_FOLLOWING_CONFIGURATION = 1010
    JOYSTICK_CONFIGURATION = 1011
    WEATHERVANING_OPTIONS = 1012


class Binary:
    sid = MsgType.EMPTY_MESSAGE
    msg_id = None
    payload = None

    def compile(self):
        if self.payload is None:
            return struct.pack('ihh', 0, self.sid, self.msg_id)
        else:
            msg = bytearray(8+len(self.payload))
            msg[:8] = struct.pack('ihh', len(self.payload), self.sid, self.msg_id.value)
            msg[8:] = self.payload
            return msg

    @staticmethod
    def parse(msg):
        try:
            length, sid, id_int = struct.unpack('ihh', msg[:8])
        except struct.error:
            raise CorruptMsgException
        except IndexError:
            raise CorruptMsgException
        try:
            msg_id = MsgType(id_int)
            if msg_id is MsgType.REMOTE_CONTROL_REQUEST_REPLY:
                return RemoteControlRequestReply(msg[8:])
            elif msg_id is MsgType.ROV_STATE:
                return RovState(msg[8:])
            else:
                raise OtherMsgTypeException
        except IndexError:
            raise CorruptMsgException


class Command(Binary):
    msg_id = MsgType.COMMAND_MESSAGE

    def __init__(self, option, sid=0):
        self.sid = sid
        self.payload = struct.pack('i', option.value)

class CommandOptions(Enum):
    NONE = 0
    START = 1
    STOP = 2
    SAVE = 3
    CLEAR_WPS = 6

class CruiseSpeed(Binary):
    msg_id = MsgType.CRUISE_SPEED

    def __init__(self, surge_speed, sway_speed=0, sid=0):
        self.sid = sid
        self.payload = struct.pack('dd', surge_speed, sway_speed)

class TrackingSpeed(Binary):
    msg_id = MsgType.TRACKING_SPEED

    def __init__(self, surge_speed, sid=0):
        self.sid = sid
        self.payload = struct.pack('d', surge_speed)

class AddWaypoints(Binary):
    msg_id = MsgType.ADD_WAYPOINTS

    def __init__(self, wp_list, sid=0):
        """

        :param wp_list: list of (N, E, D)
        :param sid:
        """
        self.sid = sid
        self.payload = bytearray(4 * (1 + 3 * len(wp_list)))
        self.payload[:4] = struct.pack('i', len(wp_list))
        for i in range(len(wp_list)):
            self.payload[4 + i*12:4 + (i+1)*12] = struct.pack('ddd', wp_list[i][0], wp_list[i][1], wp_list[i][2])
        # for i in range(len(wp_list)):
        #     print(wp_list[i][0], wp_list[i][1], wp_list[i][2])

class Setpoint(Binary):
    msg_id = MsgType.SETPOINT

    def __init__(self, setpoint, dof, absolute, sid=0):
        self.sid = sid
        if absolute:
            absolute = 1
        else:
            absolute = 0
        self.payload = struct.pack('dBB', setpoint, dof.value, absolute)

class DofOptions(Enum):
    SURGE = 1
    SWAY = 2
    HEAVE = 3
    ROLL = 4
    PITCH = 5
    YAW = 6

class PathFollowConfig(Binary):
    msg_id = MsgType.PATH_FOLLOWING_CONFIGURATION

    def __init__(self, roa, braking_zone, gain_speed, lookahead, gain_path, cornering_speed, sid=0):
        self.sid = sid
        self.payload = struct.pack('dddddd', roa, braking_zone, gain_speed, lookahead, gain_path, cornering_speed)

class GuidanceMode(Binary):
    msg_id = MsgType.SET_GUIDANCE_MODE

    def __init__(self, mode, sid=0):
        self.sid = sid
        self.payload = struct.pack('i', mode.value)

class GuidanceModeOptions(Enum):
    NONE = 0
    STATION_KEEPING = 1
    CIRCULAR_INSPECTION_MODE = 3
    PATH_FOLLOWING = 5
    CRUISE_MODE = 6

class RemoteControlRequest(Binary):
    msg_id = MsgType.REMOTE_CONTROL_REQUEST

    def __init__(self, ask_for_control, sid=0):
        self.sid = sid
        if ask_for_control:
            self.payload = struct.pack('B', 1)
        else:
            self.payload = struct.pack('B', 0)

class RemoteControlRequestReply(Binary):
    msg_id = MsgType.REMOTE_CONTROL_REQUEST_REPLY
    acquired = False

    def __init__(self, msg):
        try:
            self.token, status = struct.unpack('h?', msg)
            if self.token > 0 and status:
                self.acquired = True
        except struct.error:
            raise CorruptMsgException

class GetMessage(Binary):
    msg_id = MsgType.GET_MESSAGE

    def __init__(self, msg_ids, sid=0):
        self.sid = sid
        if msg_ids is list:
            new_list = []
            for msg_id in msg_ids:
                new_list.append(msg_id.value)
            self.payload = struct.pack('i{}h'.format(len(msg_ids)), len(msg_ids), *msg_ids)
        else:
            self.payload = struct.pack('ih', 1, msg_ids.value)

class RovState(Binary, UdpPosMsg):
    msg_id = MsgType.ROV_STATE

    def __init__(self, msg):
        try:
            self.lat, self.long, self.down, self.roll, self.pitch, self.psi = struct.unpack('6d', msg[:48])
            self.surge, self.sway, self.heave, self.roll, self.pitch, self.yaw = struct.unpack('6d', msg[48:96])
            self.a_surge, self.a_sway, self.a_heave, self.a_roll, self.a_pitch, self.a_yaw = struct.unpack('6d', msg[96:144])
            self.depth, self.alt = struct.unpack('dd', msg[144:160])
        except struct.error:
            raise CorruptMsgException
        except IndexError:
            raise CorruptMsgException