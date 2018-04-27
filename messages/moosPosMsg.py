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
    lat = 0.0
    long = 0.0

    p = 0.0
    q = 0.0
    r = 0.0
    u = 0.0
    v = 0.0
    w = 0.0

    def __init__(self, north=0, east=0, yaw=0, z=0, *kwargs):
        super().__init__(*kwargs)
        self.north = north
        self.east = east
        self.yaw = yaw
        self.z = z

    def __sub__(self, other):
        lat_diff = self.lat - other.lat
        # print('(self {}) - (other {}) = {}'.format(self.lat, other.lat, lat_diff))
        long_diff = self.long - other.long
        alpha = atan2(long_diff, lat_diff)
        dist = sqrt(lat_diff**2 + long_diff**2)
        dyaw = self.yaw - other.yaw

        dx = cos(alpha - self.yaw)*dist
        dy = sin(alpha - self.yaw)*dist
        return MoosPosMsgDiff(dx, dy, dyaw)

    # def __eq__(self, other):
    #     return (self.lat == other.lat and self.long == other.long and self.yaw == other.yaw)

    def __str__(self):
        return 'lat: {:5f},\tlong: {:5f}\t, yaw: {:5f}'.format(self.lat, self.long, self.yaw*180/pi)


if __name__ == '__main__':
    from math import pi
    new_msg = MoosPosMsg()
    new_msg.lat = -2
    new_msg.long = -2
    new_msg.yaw = 0

    old_msg = MoosPosMsg()
    old_msg.lat = 0
    old_msg.long = 0
    old_msg.yaw = pi

    res = new_msg - old_msg
    print('dx: {}\tdy: {}\tdyaw: {}'.format(res.dx, res.dy, res.dyaw*180/pi))
    diff_msg = MoosPosMsgDiff(1, 1, 0)
    print(res+diff_msg)

