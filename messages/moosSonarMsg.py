import numpy as np

from messages.sensor import Sensor


class MoosSonarMsg(Sensor):
    sensorStr = 'SONAR'
    sensor = 2
    bins = None

    bearing = 0
    step = 0
    range_scale = 0
    length = 0
    dbytes = 0
    data = 0
    time = 0
    adc8on = True
    # txNode = np.uint8(0)
    # rxNode = np.uint8(0)
    # type = np.uint8(0)
    # deviceType = np.uint8(0)
    # headStatus = np.uint8(0)
    # sweepCode = np.uint8(0)
    # hdCtrl = np.uint16(0)
    # rangeScale = np.uint16(0)
    # gain = np.uint8(0)
    # slope = np.uint16(0)
    # adSpan = np.uint8(0)
    # adLow = np.uint8(0)
    # headingOffset = np.uint16(0)
    # adInterval = np.uint16(0)
    # leftLim = np.uint16(0)
    # rightLim = np.uint16(0)
    # step = np.uint16(0)
    # bearing = np.uint16(0)
    # dataBins = np.uint16(0)
