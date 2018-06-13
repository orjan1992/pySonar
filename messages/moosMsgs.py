import pymoos as pm
from pymoos import moos_msg
import logging
from struct import unpack, calcsize
from math import pi
from ast import literal_eval
import threading

from messages.moosSonarMsg import MoosSonarMsg
from messages.moosPosMsg import MoosPosMsg
from coordinate_transformations import wrapTo2Pi
from settings import *
import cv2
from collision_avoidance.los_controller import LosController
from PyQt5.QtCore import QObject, pyqtSignal

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class MoosMsgs(QObject):
    cur_pos_msg = None
    RAD2GRAD = 3200.0/pi
    signal_new_sonar_msg = pyqtSignal(object, name='new_sonar_msg')
    cur_pos_msg = None
    cur_desired_pos_msg = None
    in_control = False
    wp_update_in_progress = threading.Lock()
    e, s = 0, 0

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

        if LosSettings.enable_los and Settings.enable_autopilot:
            self.los_stop_event = threading.Event()
            self.los_start_event = threading.Event()
            self.los_restart_event = threading.Event()
            self.los_controller = LosController(self, 0.1, self.los_stop_event, self.los_start_event, self.los_restart_event)
            self.los_thread = threading.Thread()  # threading.Thread(target=self.los_controller.loop, daemon=True)

    def run(self, host='localhost', port=9000, name='pySonar'):
        self.comms.run(host, port, name)
        self.send_msg('alt_com', 1.5)

    def close(self):
        if LosSettings.enable_los:
            self.los_stop_event.set()
            self.los_thread.join()
        self.comms.close(True)
        self.send_msg('vel_com', 0)

    def send_msg(self, var, val):
        # heading command = yaw_com
        # 'north_com'
        # 'east_com',
        # 'depth_com'
        # 'yaw_com',
        # vel_com
        if isinstance(val, str) or isinstance(val, float):
            self.comms.notify(var, val, pm.time())
        else:
            self.comms.notify(var, str(val), pm.time())

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
                sonar_msg.bearing = np.round(wrapTo2Pi(tmp[0] - 3*pi/2)*self.RAD2GRAD).astype(int)
                sonar_msg.step = round(tmp[1]*self.RAD2GRAD)
                sonar_msg.range_scale = tmp[2]
                sonar_msg.length = tmp[3]  # TODO one variable to much, which is needed?
                sonar_msg.dbytes = tmp[3]  # TODO one variable to much, which is needed?
                sonar_msg.data = tmp[4:]  # = np.array(tmp[2:])
                sonar_msg.time = msg.time()

                sonar_msg.adc8on = True
                sonar_msg.chan2 = False
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
            self.cur_pos_msg.north = msg.double()
        if msg.key() == 'currentNEDPos_y':
            self.logger_pose.debug('NEDPos y received')
            self.cur_pos_msg.east = msg.double()
        if msg.key() == 'currentNEDPos_rz':
            self.logger_pose.debug('NEDPos rz received')
            self.cur_pos_msg.yaw = msg.double()
        if msg.key() == 'currentVEHVel_r':
            self.logger_pose.debug('currentVEHVel_r received')
            self.cur_pos_msg.r = msg.double()
        if msg.key() == 'currentVEHVel_u':
            self.logger_pose.debug('currentVEHVel_u received')
            self.cur_pos_msg.u = msg.double()
        if msg.key() == 'currentVEHVel_v':
            self.logger_pose.debug('currentVEHVel_v received')
            self.cur_pos_msg.v = msg.double()
        self.los_start_event.set()
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
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentVEHVel_r')
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentVEHVel_u')
        self.comms.add_message_route_to_active_queue('pose_queue', 'currentVEHVel_v')
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
        self.comms.register('currentVEHVel_r', 0)
        self.comms.register('currentVEHVel_u', 0)
        self.comms.register('currentVEHVel_v', 0)
        self.comms.register('bins', 0)

        if Settings.collision_avoidance:
            self.comms.register('waypoints', 0)
            self.comms.register('waypoint_counter', 0)
        return True

    @threaded
    def update_wps(self, wp_list):
        if len(wp_list) < 2:
            return
        with self.wp_update_in_progress:
            if self.los_thread.isAlive():
                self.los_restart_event.set()
                self.los_stop_event.set()
                self.los_thread.join()
                self.los_stop_event.clear()
                self.los_restart_event.clear()
            self.los_controller.update_wps(wp_list)
            self.los_controller.update_pos(self.cur_pos_msg)
            self.los_thread = threading.Thread(target=self.los_controller.loop, daemon=True)
            self.los_thread.start()
            self.los_start_event.set()
            return

    @threaded
    def stop_and_turn(self, theta):
        if Settings.enable_autopilot:
            self.logger.info('Stop and turn {:.2f} deg'.format(theta * 180.0 / np.pi))
            if LosSettings.enable_los:
                if self.los_thread.isAlive():
                    self.los_stop_event.set()
                    self.los_thread.join()
                    self.los_stop_event.clear()
            else:
                self.stop_autopilot()
            self.send_msg('yaw_com', theta + self.cur_pos_msg.yaw)

    @threaded
    def stop_autopilot(self):
        self.send_msg('vel_com', 0.0)
        counter = 0
        n_set = 1
        max = 20
        surge = np.ones(max)
        surge[0] = self.cur_pos_msg.u
        acc = 1
        while abs(np.sum(surge) / n_set) > 0.02:
            # print(np.sum(v_surge)/n_set, acc)
            self.los_start_event.clear()
            self.los_start_event.wait(0.1)
            counter += 1
            if counter == max:
                counter = 0
            surge[counter] = self.cur_pos_msg.u
            acc = abs((surge[counter] - surge[counter - 1]) / 0.1)
            if n_set < max:
                n_set += 1
        return True
