from math import atan2, sqrt, cos, sin

from messages.sensor import Sensor


class MoosPosMsgDiff:
    def __init__(self, dx, dy, dpsi):
        self.dx = dx
        self.dy = dy
        self.dpsi = dpsi
        if dx > 0.01 or dy > 0.01 or dpsi > 0.000981748:
            self.big_difference = True
        else:
            self.big_difference = False

    def __add__(self, other):
        dx = self.dx + other.dx
        dy = self.dy + other.dy
        dpsi = self.dpsi + other.dpsi
        return MoosPosMsgDiff(dx, dy, dpsi)


class MoosPosMsg(Sensor):

    sensor = 1
    sensorStr = 'Position'
    id = 0
    psi = 0.0
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

    def __sub__(self, other):
        lat_diff = self.lat - other.lat
        long_diff = self.long - other.long
        alpha = atan2(long_diff, lat_diff)
        dist = sqrt(lat_diff**2 + long_diff**2)
        dpsi = self.psi - other.psi

        dx = cos(alpha - self.psi)*dist
        dy = sin(alpha - self.psi)*dist
        return MoosPosMsgDiff(dx, dy, dpsi)


if __name__ == '__main__':
    from math import pi
    new_msg = MoosPosMsg()
    new_msg.lat = 2
    new_msg.long = 2
    new_msg.psi = pi/4

    old_msg = MoosPosMsg()
    old_msg.lat = 0
    old_msg.long = 0
    old_msg.psi = 0

    res = new_msg - old_msg
    print('dx: {}\tdy: {}\tdpsi: {}'.format(res.dx, res.dy, res.dpsi*180/pi))
