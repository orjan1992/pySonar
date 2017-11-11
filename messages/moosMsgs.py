import pymoos as pm
from pymoos import moos_msg
import logging
from struct import unpack

from messages.moosSonarMsg import MoosSonarMsg
from messages.moosPosMsg import MoosPosMsg

logger = logging.getLogger('messages.MooseMsgs')


class MoosMsgs(object):
    cur_pos_msg = None
    new_pos_msg_func = None
    new_sonar_msg_func = None
    pos_msg_flags = [False, False, False]

    def __init__(self, host='localhost', port=9000, name='pySonar'):
        """
        :param host: MOOOSE host name/ip
        :param port: MOOSE port
        :param name: Name of program
        """
        logger.debug('MOOse msgs initiated')
        self.comms = pm.comms()
        self.comms.set_on_connect_callback(self.on_connect)
        self.add_queues()
        self.comms.run(host, port, name)
        self.cur_pos_msg = MoosPosMsg()

    def set_on_pos_msg_callback(self, cb):
        self.new_pos_msg_func = cb

    def set_on_sonar_msg_callback(self, cb):
        self.new_sonar_msg_func = cb


    def on_connect(self):
        self.comms.register('currentNEDPos_rz', 0)
        self.comms.register('currentNEDPos_x', 0)
        self.comms.register('currentNEDPos_y', 0)
        self.comms.register('bins', 0)
        return True

    def bins_queue(self, msg):
        if msg.is_binary():
            sonar_msg = MoosSonarMsg()
            sonar_msg.bearing, sonar_msg.length = unpack('<dH', msg.binary_data()[0:10])
            sonar_msg.bins = unpack('<{}f'.format(sonar_msg.length), msg.binary_data()[10:])
            self.new_sonar_msg_func(sonar_msg)
        else:
            print('tset')
        print('dfgdfgdfg')
        return True

    def pose_queue(self, msg):
        # TODO implement time
        # a = moos_msg()
        # a.
        # print(msg.name())
        # print(msg.key())
        if msg.key() == 'currentNEDPos_x':
            self.cur_pos_msg.x = msg.double()
            self.cur_pos_msg.moos_flag_lat = True
            self.pos_msg_flags[0] = True
        if msg.key() == 'currentNEDPos_y':
            self.cur_pos_msg.y = msg.double()
            self.cur_pos_msg.moos_flag_long = True
            self.pos_msg_flags[1] = True
        if msg.key() == 'currentNEDPos_rz':
            self.cur_pos_msg.rot = msg.double()
            self.pos_msg_flags[2] = True
        if all(self.pos_msg_flags):
            self.new_pos_msg_func(self.cur_pos_msg)
            self.cur_pos_msg = MoosPosMsg()
            self.pos_msg_flags = [False, False, False]
        return True

    def add_queues(self):
        # self.comms.add_active_queue('pose_queue', self.pose_queue)
        # self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_rz')
        # self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_x')
        # self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_y')
        self.comms.add_active_queue('bins_queue', self.bins_queue)
        self.comms.add_message_route_to_active_queue('bins_queue', 'bins')