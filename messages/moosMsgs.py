import pymoos as pm
from pymoos import moos_msg
import logging
from struct import unpack, error as struct_error, calcsize

from messages.moosSonarMsg import MoosSonarMsg
from messages.moosPosMsg import MoosPosMsg


class MoosMsgs(object):
    cur_pos_msg = None
    pos_msg_flags = [False, False, False]

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
        self.logger_pose.disabled = True
        self.logger_pose.disabled = True
        self.logger.disabled = True
        self.logger.debug('MOOS msgs initiated')

        # Init
        self.comms = pm.comms()
        self.comms.set_on_connect_callback(self.on_connect)
        self.add_queues()

        self.cur_pos_msg = MoosPosMsg()

        self.new_pos_msg_func = self.dummy_func
        self.new_sonar_msg_func = self.dummy_func

    def run(self, host='localhost', port=9000, name='pySonar'):
        self.comms.run(host, port, name)

    def close(self):
        self.comms.close(True)

    def set_on_pos_msg_callback(self, cb):
        self.new_pos_msg_func = cb

    def set_on_sonar_msg_callback(self, cb):
        self.new_sonar_msg_func = cb

    def bins_queue(self, msg):
        self.logger_bins.debug('Message received of type: {}'.format(type(msg)))
        if msg.is_binary():
            sonar_msg = MoosSonarMsg()
            self.logger_bins.debug('Binary message. length: {}\t calcsize: {}'.format(msg.binary_data_size(),
                                                                                      calcsize('<dH{:d}f'.format((msg.binary_data_size()-10)//4))))
            self.logger_bins.debug('time: {}'.format(msg.time()))
            self.logger_bins.debug(msg.is_binary())
            self.logger_bins.debug(type(msg.binary_data()))
            data = msg.binary_data().encode('latin-1')
            tmp = unpack('>dH{:d}f'.format((len(data) - 10) // 4), data)
            self.logger_bins.debug('Unpacking complte')
            sonar_msg.bearing = tmp[0]
            sonar_msg.length = tmp[1]
            sonar_msg.data = tmp[2:] # = np.array(tmp[2:])
            self.new_sonar_msg_func(sonar_msg)
            self.logger_bins.debug('Callback OK')
        return True


    def pose_queue(self, msg):
        # TODO implement time
        self.logger_pose.debug('Message recieved. Type{}'.format(type(msg)))
        if msg.key() == 'currentNEDPos_x':
            self.logger_pose.debug('NEDPos x received')
            self.cur_pos_msg.x = msg.double()
            self.cur_pos_msg.moos_flag_lat = True
            self.pos_msg_flags[0] = True
        if msg.key() == 'currentNEDPos_y':
            self.logger_pose.debug('NEDPos y received')
            self.cur_pos_msg.y = msg.double()
            self.cur_pos_msg.moos_flag_long = True
            self.pos_msg_flags[1] = True
        if msg.key() == 'currentNEDPos_rz':
            self.logger_pose.debug('NEDPos rz received')
            self.cur_pos_msg.head = msg.double()
            self.pos_msg_flags[2] = True
        if all(self.pos_msg_flags):
            self.logger_pose.debug('Complete message received')
            if self.new_pos_msg_func(self.cur_pos_msg):
                self.logger_pose.debug('Callback complete')

            self.cur_pos_msg = MoosPosMsg()
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

    def dummy_func(self, msg):
        # inital function for callbacks
        return True