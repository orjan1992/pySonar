from readLogFile.sensor import Sensor
import numpy as np
class SonarMsg(Sensor):
    sensorStr = 'SONAR'
    sensor = 2

    txNode = np.uint8(0)
    rxNode = np.uint8(0)
    type = np.uint8(0)
    deviceType = np.uint8(0)
    headStatus = np.uint8(0)
    sweepCode = np.uint8(0)
    hdCtrl = np.uint16(0)
    rangeScale = np.uint16(0)
    gain = np.uint8(0)
    slope = np.uint16(0)
    adSpan = np.uint8(0)
    adLow = np.uint8(0)
    headingOffset = np.uint16(0)
    adInterval = np.uint16(0)
    leftLim = np.uint16(0)
    rightLim = np.uint16(0)
    step = np.uint16(0)
    bearing = np.uint16(0)
    dataBins = np.uint16(0)
    time = np.double(0)