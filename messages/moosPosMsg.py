from math import cos, sin

from messages.sensor import Sensor


class MoosPosMsg(Sensor):

    sensor = 1
    sensorStr = 'Position'
    id = 0
    head = 0.0
    roll = 0.0
    pitch = 0.0
    depth = 0.0
    alt = 0.0

    def _get_x(self):
        x = self.x*cos(self.head) - self*ysin(self.head)
        return x
    def _set_x(self):
        x = cos(self.head) - sin(self.head)
        return x

    x = property(_get_x, _set_x)

    def _get_y(self):
        y = sin(self.head) + cos(self.head)
        return y

    y = property(_get_y)
