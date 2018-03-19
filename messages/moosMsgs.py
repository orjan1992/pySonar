import pymoos as pm
from pymoos import moos_msg
import logging
from struct import unpack, calcsize
from math import pi
from ast import literal_eval

from messages.moosSonarMsg import MoosSonarMsg
from messages.moosPosMsg import MoosPosMsg
from settings import *
import cv2
from PyQt5.QtCore import QObject, pyqtSignal


class MoosMsgs(QObject):
    cur_pos_msg = None
    RAD2GRAD = 3200.0/pi
    signal_new_wp = pyqtSignal(object, name='new_wp')
    signal_new_sonar_msg = pyqtSignal(object, name='new_sonar_msg')

    def __init__(self, sonar_msg_callback=None, waypoints_callback=None):
        """
        :param host: MOOS host name/ip
        :param port: MOOS port
        :param name: Name of program
        """
        # Logger stuff
        super().__init__()
        self.logger = logging.getLogger('messages.MoosMsgs')
        self.logger_bins = logging.getLogger('messages.MoosMsgs.bins')
        self.logger_pose = logging.getLogger('messages.MoosMsgs.pose')
        self.logger.debug('MOOS msgs initiated')

        # Init
        self.comms = pm.comms()
        self.comms.set_on_connect_callback(self.on_connect)
        self.add_queues()

        self.cur_pos_msg = MoosPosMsg()

        self.waypoint_list = None
        self.waypoint_counter = None

    def set_waypoints_callback(self, waypoints_callback):
        self.waypoints_callback = waypoints_callback

    def run(self, host='localhost', port=9000, name='pySonar'):
        self.comms.run(host, port, name)

    def close(self):
        self.comms.close(True)

    def send_msg(self, var, val):
        self.comms.notify(var, val, pm.time())

    def bins_queue(self, msg):
        try:
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
                sonar_msg.data = tmp[4:]  # = np.array(tmp[2:])
                sonar_msg.time = msg.time()

                sonar_msg.adc8on = True
                sonar_msg.chan2 = True
                # self.sonar_callback(sonar_msg)
                self.signal_new_sonar_msg.emit(sonar_msg)
                self.logger_bins.debug('Callback OK')
        except Exception as e:
            print(e)
        return True

    def pose_queue(self, msg):
        self.logger_pose.debug('Message recieved. Type{}'.format(type(msg)))
        if msg.key() == 'currentNEDPos_x':
            self.logger_pose.debug('NEDPos x received')
            self.cur_pos_msg.lat = msg.double()
        if msg.key() == 'currentNEDPos_y':
            self.logger_pose.debug('NEDPos y received')
            self.cur_pos_msg.long = msg.double()
        if msg.key() == 'currentNEDPos_rz':
            self.logger_pose.debug('NEDPos rz received')
            self.cur_pos_msg.psi = msg.double()
        return True

    def waypoints_queue(self, msg):
        try:
            if msg.key() == 'waypoint_counter':
                self.waypoint_counter = int(msg.double())
                self.signal_new_wp.emit(self.waypoint_counter)
            else:
                self.waypoint_list = literal_eval(msg.string())
                self.signal_new_wp.emit(self.waypoint_list)
        except Exception as e:
            print(e)
        return True

    def add_queues(self):
        self.logger.debug('Add queues running')
        self.comms.add_active_queue('pose_queue', self.pose_queue)
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_rz')
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_x')
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentNEDPos_y')
        self.comms.add_active_queue('bins_queue', self.bins_queue)
        self.comms.add_message_route_to_active_queue('bins_queue', 'bins')
        if Settings.collision_avoidance:
            self.comms.add_active_queue('waypoints_queue', self.waypoints_queue)
            self.comms.add_message_route_to_active_queue('waypoints_queue', 'waypoints')
            self.comms.add_message_route_to_active_queue('waypoints_queue', 'waypoint_counter')
        return True

    def on_connect(self):
        self.logger.debug('On connect running')
        self.comms.register('currentNEDPos_rz', 0)
        self.comms.register('currentNEDPos_x', 0)
        self.comms.register('currentNEDPos_y', 0)
        self.comms.register('bins', 0)

        if Settings.collision_avoidance:
            self.comms.register('waypoints', 0)
            self.comms.register('waypoint_counter', 0)
        return True
