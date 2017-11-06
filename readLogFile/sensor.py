class Sensor(object):
    sensorStr = 'None'
    sensor = 0

    def __init__(self, timeStr):
        if not timeStr:
            self.timeStr = ''
        else:
            self.timeStr = timeStr
