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
    x = 0.0
    y = 0.0
