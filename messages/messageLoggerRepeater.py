import pymoos as pm
import csv
import time
import select
import sys
import struct
from math import pi
from datetime import datetime, timedelta
from time import sleep
import numpy as np



class MoosLoggerRepeater:
    RAD2GRAD = 3200.0 / pi

    def __init__(self, fname):
        # Init
        self.comms = pm.comms()
        self.fname = fname
        self.f = open(fname, 'r')
        self.reader = csv.DictReader(self.f, fieldnames=['time', 'key', 'type',
                                                         'bearing', 'step', 'range_scale',
                                                         'length', 'dbytes', 'data'])

    def run(self, host='localhost', port=9000, name='logger'):
        self.comms.run(host, port, name)
        first_row = True
        counter = 0
        for row in self.reader:
            time = timedelta(seconds=float(row['time']))
            if first_row:
                self.now = datetime.now()
                self.first_time = time
                first_row = False
            counter += 1
            print(counter)
            while datetime.now() < self.now + time - self.first_time:
                sleep(0.000001)
            if row['type'] == 'binary':
                data = np.fromstring(row['data'][1:-1], dtype=np.uint8, sep=',')
                msg = struct.pack('>dddH{}B'.format(len(data)), float(row['bearing']),
                                  float(row['step']), float(row['range_scale']),
                                  len(data), *data)
                self.comms.notify_binary(row['key'], msg.decode('latin-1'), pm.time())
            else:
                self.comms.notify(row['key'], float(row['data']), pm.time())


    def close(self):
        self.comms.close(True)
        self.f.close()


if __name__ == '__main__':
    logger = MoosLoggerRepeater('log.csv')
    logger.run()
    logger.close()
