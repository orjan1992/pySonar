import pymoos as pm
from pymoos import moos_msg
import logging
from struct import unpack, calcsize
from math import pi
from blinker import signal

from messages.moosSonarMsg import MoosSonarMsg
from messages.posMsg import PosMsg


class MoosMsgs(object):
    cur_pos_msg = None
    pos_msg_flags = [False, False, False]
    RAD2GRAD = 3200/pi

    def __init__(self):
        """
        :param host: MOOS host name/ip
        :param port: MOOS port
        :param name: Name of program
        """
        # Logger stuff
        self.logger = logging.getLogger('messages.MoosMsgs')
        self.logger_bins = logging.getLogger('messages.MoosMsgs.bins')
        self.logger_pose = logging.getLogger('messages.MoosMsgs.pose')
        self.logger.debug('MOOS msgs initiated')

        # Init
        self.comms = pm.comms()
        self.comms.set_on_connect_callback(self.on_connect)
        self.add_queues()

        self.cur_pos_msg = PosMsg()
        self.new_msg_signal = signal('new_msg_sonar')
        self.new_pos_msg_signal = signal('new_msg_pos')

    def run(self, host='localhost', port=9000, name='pySonar'):
        self.comms.run(host, port, name)

    def close(self):
        self.comms.close(True)

    def bins_queue(self, msg):
        # TODO implement rangescale step osv
        self.logger_bins.debug('Message received of type: {}'.format(type(msg)))
        if msg.is_binary():
            sonar_msg = MoosSonarMsg()
            self.logger_bins.debug('Binary message. length: {}\t calcsize: {}'.format(msg.binary_data_size(),
                                                                                      calcsize('<dH{:d}f'.format((msg.binary_data_size()-10)//4))))
            self.logger_bins.debug('time: {}'.format(msg.time()))
            self.logger_bins.debug(msg.is_binary())
            self.logger_bins.debug(type(msg.binary_data()))
            data = msg.binary_data().encode('latin-1')
            tmp = unpack('>dddH{:d}B'.format((len(data) - 26)), data)
            self.logger_bins.debug('Unpacking complte')
            sonar_msg.bearing = round((tmp[0] + pi/2)*self.RAD2GRAD)
            sonar_msg.step = round(tmp[1]*self.RAD2GRAD)
            sonar_msg.range_scale = tmp[2]
            sonar_msg.length = tmp[3]  # TODO one variable to much, which is needed?
            sonar_msg.dbytes = tmp[3]  # TODO one variable to much, which is needed?
            sonar_msg.data = tmp[4:] # = np.array(tmp[2:])
            sonar_msg.time = msg.time()

            sonar_msg.adc8on = True
            sonar_msg.chan2 = True
            try:
                self.new_msg_signal.send(self, msg=sonar_msg)
            except Exception as err:
                print("{0}".format(err))
            self.logger_bins.debug('Callback OK')
        return True


    def pose_queue(self, msg):
        # TODO implement time
        self.logger_pose.debug('Message recieved. Type{}'.format(type(msg)))
        if msg.key() == 'currentNEDPos_x':
            self.logger_pose.debug('NEDPos x received')
            self.cur_pos_msg.lat = msg.double()
            self.cur_pos_msg.moos_flag_lat = True
            self.pos_msg_flags[0] = True
        if msg.key() == 'currentNEDPos_y':
            self.logger_pose.debug('NEDPos y received')
            self.cur_pos_msg.long = msg.double()
            self.cur_pos_msg.moos_flag_long = True
            self.pos_msg_flags[1] = True
        if msg.key() == 'currentNEDPos_rz':
            self.logger_pose.debug('NEDPos rz received')
            self.cur_pos_msg.psi = msg.double()
            self.pos_msg_flags[2] = True
        if all(self.pos_msg_flags):
            try:
                self.new_pos_msg_signal.send(self, msg=self.cur_pos_msg)
            except Exception as err:
                print("{0}".format(err))
            self.pos_msg_flags = [False, False, False]
        return True

    def add_queues(self):
        self.logger.debug('Add queues running')
        self.comms.add_active_queue('pose_queue', self.pose_queue)
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_rz')
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_x')
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_y')
        self.comms.add_active_queue('bins_queue', self.bins_queue)
        self.comms.add_message_route_to_active_queue('bins_queue', 'bins')
        return True

    def on_connect(self):
        self.logger.debug('On connect running')
        self.comms.register('currentNEDPos_rz', 0)
        self.comms.register('currentNEDPos_x', 0)
        self.comms.register('currentNEDPos_y', 0)
        self.comms.register('bins', 0)
        return True
