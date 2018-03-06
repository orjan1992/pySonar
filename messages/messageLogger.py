import pymoos as pm
import csv
from time import sleep
import select
import sys
import struct
from math import pi


class MoosLogger:
    RAD2GRAD = 3200.0 / pi

    def __init__(self, fname):
        # Init
        self.comms = pm.comms()
        self.comms.set_on_connect_callback(self.on_connect)
        self.add_queues()
        self.fname = fname
        self.f = open(fname, 'w')
        self.writer = csv.DictWriter(self.f, fieldnames=['time', 'key', 'type',
                                                         'bearing', 'step', 'range_scale',
                                                         'length', 'dbytes', 'data'])

    def run(self, host='localhost', port=9000, name='logger'):
        self.comms.run(host, port, name)

    def close(self):
        self.comms.close(True)
        self.f.close()

    def on_connect(self):
        self.comms.register('currentNEDPos_rz', 0)
        self.comms.register('currentNEDPos_x', 0)
        self.comms.register('currentNEDPos_y', 0)
        self.comms.register('bins', 0)
        return True

    def add_queues(self):
        self.comms.add_active_queue('msg_queue', self.msg_queue)
        self.comms.add_message_route_to_active_queue('msg_queue', 'currentNEDPos_rz')
        self.comms.add_message_route_to_active_queue('msg_queue', 'currentNEDPos_x')
        self.comms.add_message_route_to_active_queue('msg_queue', 'currentNEDPos_y')
        self.comms.add_message_route_to_active_queue('msg_queue', 'bins')
        return True

    def msg_queue(self, msg):
        if msg.is_binary():
            data = msg.binary_data().encode('latin-1')
            tmp = struct.unpack('>dddH{:d}B'.format((len(data) - 26)), data)
            bearing = tmp[0]
            step = tmp[1]
            range_scale = tmp[2]
            length = tmp[3]
            dbytes = tmp[3]
            data = tmp[4:]

            self.writer.writerow({'time': msg.time(), 'key': msg.key(), 'type': 'binary',
                             'bearing': bearing, 'step': step, 'range_scale': range_scale,
                                      'length': length, 'dbytes': dbytes, 'data': data})
        else:
            self.writer.writerow({'time': msg.time(), 'key': msg.key(), 'type': 'double', 'data': msg.double()})
        return True


if __name__ == '__main__':
    logger = MoosLogger('log_still.csv')
    logger.run()
    while not select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        sleep(0.005)
    logger.close()
