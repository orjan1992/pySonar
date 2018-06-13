from enum import Enum
import struct
from messages.udpMsg import CorruptMsgException, OtherMsgTypeException, UdpPosMsg
from numpy import arctan2, sin, cos
import logging
from time import strftime
logger = logging.getLogger('AutopilotMsg')

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
            logger.warning('Could not parse msg header: {}'.format(msg))
            raise CorruptMsgException
        except IndexError:
            logger.warning('Could not parse msg header: {}'.format(msg))
            raise CorruptMsgException
        try:
            msg_id = MsgType(id_int)
            if msg_id is MsgType.REMOTE_CONTROL_REQUEST_REPLY:
                return RemoteControlRequestReply(msg[8:])
            elif msg_id is MsgType.ROV_STATE:
                return RovState(msg[8:])
            elif msg_id is MsgType.ROV_STATE_DESIRED:
                return RovState(msg[8:], msg_id)
            elif msg_id is MsgType.ERROR:
                return Error(msg[8:])
            elif msg_id is MsgType.WARNING_GUIDANCE:
                return WarningGuidance(msg[8:])
            else:
                logger.info('Unknown msg from autopilot server, msg_id: '.format(id_int))
                raise OtherMsgTypeException
        except IndexError:
            logger.warning('Could not parse msg: length: {}, sid: {}, id_int: {}'.format(length, sid, id_int))
            raise CorruptMsgException()
        except CorruptMsgException:
            logger.warning('Could not parse msg: length: {}, sid: {}, id_int: {}'.format(length, sid, id_int))
            raise CorruptMsgException()


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
            self.payload[4 + i*24:4 + (i+1)*24] = struct.pack('ddd', wp_list[i][0], wp_list[i][1], wp_list[i][2])
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

class Dofs(Enum):
    SURGE = 0
    NORTH = 0
    SWAY = 1
    EAST = 1
    HEAVE = 2
    DEPTH = 2
    ALTITUDE = 2
    ROLL = 3
    PITCH = 4
    YAW = 5

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
        except struct.error as e:
            raise CorruptMsgException

class GetMessage(Binary):
    msg_id = MsgType.GET_MESSAGE

    def __init__(self, msg_ids, sid=0):
        self.sid = sid
        try:
            new_list = []
            for msg_id in msg_ids:
                new_list.append(msg_id.value)
            self.payload = struct.pack('i{}h'.format(len(msg_ids)), len(msg_ids), *new_list)
        except TypeError:
            self.payload = struct.pack('ih', 1, msg_ids.value)

class Tuning(Binary):
    msg_id = MsgType.CONTROLLER_TUNING

    def __init__(self, dof, step, sid=0):
        self.sid = sid
        self.payload = struct.pack('BdB', dof.value, step, 1)

class VerticalPos(Binary):
    msg_id = MsgType.SETPOINT_VERTICAL

    def __init__(self, type, setpoint=None, sid=0):
        self.sid = sid
        if setpoint is None:
            self.payload = struct.pack('BB', type.value, 0)
        else:
            self.payload = struct.pack('BBd', type.value, 1, setpoint)

class ControllerOptions(Binary):
    msg_id = MsgType.CONTROLLER_OPTIONS

    def __init__(self, enabled_dofs, enable_output=True, enable_performance=False, sid=0):
        self.sid = sid
        enabled = 0
        try:
            for dof in enabled_dofs:
                if dof is Dofs.SURGE:
                    enabled |= 1
                elif dof is Dofs.SWAY:
                    enabled |= 2
                elif dof is Dofs.HEAVE:
                    enabled |= 4
                elif dof is Dofs.ROLL:
                    enabled |= 8
                elif dof is Dofs.PITCH:
                    enabled |= 16
                elif dof is Dofs.YAW:
                    enabled |= 32
        except TypeError:
            pass
        self.payload = struct.pack('iBB', enabled, enable_output, enable_performance)

class Error(Binary):
    msg_id = MsgType.ERROR

    def __init__(self, msg):
        try:
            self.error = ErrorCode(struct.unpack('i', msg))
        except struct.error:
            raise CorruptMsgException

    def __str__(self):
        return 'Autopilot error: {}'.format(self.error.name)

class ErrorCode(Enum):
    NO_ERROR = 0
    PLC_COM = 1
    PAR_FILE_LOAD = 2
    CONTROL_SYS = 3
    STOP_THREAD = 4
    INVALID_OPERATION = 5
    LOST_CONNECTION = 9

class WarningGuidance(Binary):
    msg_id = MsgType.WARNING_GUIDANCE

    def __init__(self, msg):
        try:
            warning_code, self.is_active = struct.unpack('I?', msg)
        except struct.error:
            raise CorruptMsgException
        self.north = (1 & warning_code) == 1
        self.east = (2 & warning_code) == 2
        self.vert = (4 & warning_code) == 4
        self.heading = (8 & warning_code) == 8
        self.path = (16 & warning_code) == 16

    def __str__(self):
        error_list = []
        if self.north:
            error_list.append('Guidance Warning: North deviation!')
        if self.east:
            error_list.append('Guidance Warning: East deviation!')
        if self.vert:
            error_list.append('Guidance Warning: Vertical deviation!')
        if self.heading:
            error_list.append('Guidance Warning: Heading deviation!')
        if self.path:
            error_list.append('Guidance Warning: Path deviation!')
        return str(error_list)


class VerticalPosOptions(Enum):
    ALTITUDE = 0
    DEPTH = 1

class RovState(Binary):
    def __init__(self, msg, msg_id=MsgType.ROV_STATE):
        self.msg_id = msg_id
        try:
            self.north, self.east, self.down, self.roll, self.pitch, self.yaw = struct.unpack('6d', msg[:48])
            self.v_surge, self.v_sway, self.v_heave, self.roll, self.pitch, self.v_yaw = struct.unpack('6d', msg[48:96])
            self.a_surge, self.a_sway, self.a_heave, self.a_roll, self.a_pitch, self.a_yaw = struct.unpack('6d', msg[96:144])
            self.depth, self.alt = struct.unpack('dd', msg[144:160])
        except struct.error:
            raise CorruptMsgException
        except IndexError:
            raise CorruptMsgException

    def __sub__(self, other):
        try:
            north_diff = self.north - other.north
        except AttributeError:
            a=1
        # print('(self {}) - (other {}) = {}'.format(self.north, other.north, north_diff))
        east_diff = self.east - other.east
        alpha = arctan2(east_diff, north_diff)
        dist = (north_diff ** 2 + east_diff ** 2)**0.5
        dyaw = self.yaw - other.yaw

        dx = cos(alpha - self.yaw) * dist
        dy = sin(alpha - self.yaw) * dist
        return RovStateDiff(dx, dy, dyaw, self.v_surge - other.v_surge, self.v_sway - other.v_sway)

    def __str__(self):
        return 'yaw: {}, roll: {}, pitch: {}, alt: {}, north: {}, east: {}'.format(self.yaw, self.roll, self.pitch,
                                                                                   self.alt, self.north, self.east)

    def to_tuple(self):
        return self.north, self.east, self.down, self.yaw, self.v_surge, self.v_sway, self.v_heave, self.v_yaw, strftime("%Y%m%d-%H%M%S")

class RovStateDiff:
    def __init__(self, dx, dy, dyaw, d_surge, d_sway):
        self.dx = dx
        self.dy = dy
        self.dyaw = dyaw
        self.surge = d_surge
        self.sway = d_sway

    def is_small(self, los=False):
        absolute = abs(self)
        if los:
            return absolute.dyaw < 0.035 and absolute.surge < 0.1
        else:
            return absolute.dx < 0.1 and absolute.dy < 0.1 and absolute.dyaw < 0.035 and absolute.surge < 0.1

    def __add__(self, other):
        self.dx += other.dx
        self.dy += other.dy
        self.dyaw += other.dyaw
        self.surge += other.surge
        self.sway += other.sway

    def __sub__(self, other):
        self.dx -= other.dx
        self.dy -= other.dy
        self.dyaw -= other.dyaw
        self.surge -= other.surge
        self.sway -= other.sway

    def __abs__(self):
        return RovStateDiff(abs(self.dx), abs(self.dy), abs(self.dyaw), abs(self.surge), abs(self.sway))

    def __str__(self):
        return 'dx: {},\tdy: {}\t, dyaw: {}'.format(self.dx, self.dy, self.dyaw * 180 / pi)