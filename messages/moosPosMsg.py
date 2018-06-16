from math import atan2, sqrt, cos, sin, pi
import logging
from messages.sensor import Sensor
from settings import Map

logger = logging.getLogger('moosPosMsg')

class MoosPosMsgDiff:
    def __init__(self, dx, dy, dyaw):
        self.dx = dx
        self.dy = dy
        self.dyaw = dyaw

    def __add__(self, other):
        _dx = self.dx + other.dx
        _dy = self.dy + other.dy
        _dyaw = self.dyaw + other.dyaw
        logger.debug('self={}, other={}, sum={}'.format(self.dx, other.dx, _dx))
        return MoosPosMsgDiff(_dx, _dy, _dyaw)

    def __str__(self):
        return 'dx: {},\tdy: {}\t, dyaw: {}'.format(self.dx, self.dy, self.dyaw*180/pi)


class MoosPosMsg(Sensor):

    sensor = 1
    sensorStr = 'Position'
    id = 0
    yaw = 0.0
    roll = 0.0
    pitch = 0.0
    depth = 0.0
    z = 0.0
    alt = 0.0

    p = 0.0
    q = 0.0
    r = 0.0
    u = 0.0
    v = 0.0
    w = 0.0

    psi_ref = 0.0
    u_ref = 0.0
    z_ref = 0.0

    def __init__(self, north=0, east=0, yaw=0, z=0, alt=0, *kwargs):
        super().__init__(*kwargs)
        self.north = north
        self.east = east
        self.yaw = yaw
        self.z = z
        self.alt = alt

    def __sub__(self, other):
        north_diff = self.north - other.north
        # print('(self {}) - (other {}) = {}'.format(self.north, other.north, north_diff))
        east_diff = self.east - other.east
        alpha = atan2(east_diff, north_diff)
        dist = sqrt(north_diff**2 + east_diff**2)
        dyaw = self.yaw - other.yaw

        dx = cos(alpha - self.yaw)*dist
        dy = sin(alpha - self.yaw)*dist
        return MoosPosMsgDiff(dx, dy, dyaw)

    # def __eq__(self, other):
    #     return (self.north == other.north and self.east == other.east and self.yaw == other.yaw)

    def __str__(self):
        return 'north: {:5f},\teast: {:5f}\t, yaw: {:5f}'.format(self.north, self.east, self.yaw*180/pi)

    def to_tuple(self):
        return self.north, self.east, self.alt, self.yaw, self.u, self.v, self.r, self.psi_ref, self.u_ref, self.z_ref


if __name__ == '__main__':
    from math import pi
    new_msg = MoosPosMsg()
    new_msg.north = -2
    new_msg.east = -2
    new_msg.yaw = 0

    old_msg = MoosPosMsg()
    old_msg.north = 0
    old_msg.east = 0
    old_msg.yaw = pi

    res = new_msg - old_msg
    print('dx: {}\tdy: {}\tdyaw: {}'.format(res.dx, res.dy, res.dyaw*180/pi))
    diff_msg = MoosPosMsgDiff(1, 1, 0)
    print(res+diff_msg)

