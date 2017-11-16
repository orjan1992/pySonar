from math import cos, sin, pi
import numpy as np
from messages.sensor import Sensor
from readLogFile.helperFunctions import Wrap2pi
import logging
logger = logging.getLogger('messages.posMsg')
import csv

class PosMsg(Sensor):

    sensor = 1
    sensorStr = 'Position'
    id = 0
    head = 0.0
    roll = 0.0
    pitch = 0.0
    depth = 0.0
    alt = 0.0
    lat = 0.0
    long = 0.0

    # def _get_x(self):
    #     x = self.lat*cos(self.head) - self.long*sin(self.head)
    #     return self.lat
    #
    # x = property(_get_x)
    #
    # def _get_y(self):
    #     y = self.lat*sin(self.head) + self.long*cos(self.head)
    #     return self.long
    #
    # y = property(_get_y)

    def __sub__(self, other):
        msg = PosMsg()
        R = np.array([[cos(self.head), -sin(self.head)], [sin(self.head), cos(self.head)]])
        lat_long = np.array([[(self.lat - other.lat)], [(self.long - other.long)]])
        xy = np.dot(R.T, lat_long)
        msg.x = xy[0]
        msg.y = xy[1]
        msg.head = Wrap2pi(self.head - other.head)
        return msg

    def write_to_csv(self, writer):
        writer.writerow([self.time, self.lat, self.long])
