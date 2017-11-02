class PosMsg(object):
    type = 1
    def __init__(self, time):
        setattr(self, 'time', time)