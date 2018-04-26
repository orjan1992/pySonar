from settings import CollisionSettings, LosSettings
import threading
import messages.AutoPilotMsg as ap
import logging
import numpy as np
from coordinate_transformations import wrapToPi, wrapTo2Pi
from collision_avoidance.path_smoothing import path_grad
logger = logging.getLogger('LosController')

def get_angle(wp1, wp2):
    return np.arctan2((wp2[1] - wp1[1]), (wp2[0] - wp1[0]))

def get_errors(wp, pos, alpha):
    s = (pos[0] - wp[0])*np.cos(alpha) + (pos[1] - wp[1])*np.sin(alpha)
    e = -(pos[0] - wp[0])*np.sin(alpha) + (pos[1] - wp[1])*np.cos(alpha)
    return e, s

def segment_length(wp1, wp2):
    return ((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)**0.5


class LosController:
    wp_list = []
    wp_grad = []
    wp_counter = 0
    pos_msg = None
    last_chi = 0
    counter = 0
    e = 0
    delta = 0
    turn_velocity = 0
    roa = LosSettings.roa

    def __init__(self, msg_client, timeout, stopped_event=threading.Event(), start_event=threading.Event()):
        self.normal_surge_speed = LosSettings.cruise_speed
        self.surge_speed = self.normal_surge_speed
        self.msg_client = msg_client
        self.timeout = timeout
        self.stopped_event = stopped_event
        self.start_event = start_event
        self.lock = threading.Lock()
        self.new_waypoints = True

    def get_info(self):
        with self.lock:
            self.start_event.clear()
            return self.pos_msg,  self.wp_list, self.wp_counter, self.wp_grad

    def set_info(self, pos_msg, wp_list, wp_counter, new_wayponts):
        with self.lock:
            self.pos_msg = pos_msg
            self.wp_list = wp_list
            self.wp_counter = wp_counter
            self.new_waypoints = new_wayponts

    def get_los_values(self, wp1, wp2, pos):
        alpha = get_angle(wp1, wp2)
        e, s = get_errors(wp1, (pos.lat, pos.long), alpha)
        delta = segment_length(wp1, wp2) - s
        chi_r = np.arctan(-e / (self.surge_speed*LosSettings.look_ahead_time))
        # print(chi_r*180.0/np.pi)
        chi = chi_r + alpha
        return wrapTo2Pi(chi), delta, e

    def turn_vel(self, i):
        if self.wp_grad[i] == 0:
            return self.normal_surge_speed, -1
        self.turn_velocity = min(self.normal_surge_speed, LosSettings.safe_turning_speed/(2*self.wp_grad[i]))
        diff = self.surge_speed - self.turn_velocity
        dist = -1
        if diff > 0:
            t = diff / LosSettings.max_acc
            dist = self.surge_speed*t + 0.5*LosSettings.max_acc*t**2
        return self.turn_velocity, dist

    def set_speed(self, speed):
        if speed != self.surge_speed:
            self.surge_speed = speed
            self.msg_client.send_autopilot_msg(ap.CruiseSpeed(speed))

    def loop(self):
        if len(self.wp_list) < 2 or self.pos_msg is None:
            self.start_event.wait()

        # Initial check
        pos_msg, wp_list, wp_counter, wp_grad = self.get_info()
        chi = wrapTo2Pi(get_angle(wp_list[0], wp_list[1]))
        if segment_length(wp_list[0], (pos_msg.lat, pos_msg.long)) > LosSettings.roa or \
                abs(chi - pos_msg.psi) > LosSettings.start_heading_diff:
            self.msg_client.switch_ap_mode(ap.GuidanceModeOptions.STATION_KEEPING)
            self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][0], ap.Dofs.NORTH, True))
            self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][1], ap.Dofs.EAST, True))
            self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][2], ap.Dofs.DEPTH, True))
            self.msg_client.send_autopilot_msg(ap.Setpoint(chi, ap.Dofs.YAW, True))
            logger.info('To far from inital setpoint. Moving to: '
                        '[N={:.6f}, E={:.6f}, D={:.2f}, YAW={:.2f} deg]'.format(wp_list[0][0], wp_list[0][1],
                                                                  wp_list[0][2], chi*180.0/np.pi))

            while self.start_event.wait() and not self.stopped_event.isSet() and \
                    (segment_length(wp_list[0], (pos_msg.lat, pos_msg.long)) > 1 or
                            abs(chi - pos_msg.psi) > LosSettings.start_heading_diff):
                pos_msg, wp_list, wp_counter, wp_grad = self.get_info()

        logger.info('Switching to Cruise Mode')
        self.msg_client.switch_ap_mode(ap.GuidanceModeOptions.CRUISE_MODE)
        turn_speed, slow_down_dist = self.turn_vel(0)
        self.set_speed(turn_speed)
        logger.info('Setting cruise speed: {} m/s'.format(self.surge_speed))
        self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][2], ap.Dofs.DEPTH, True))
        self.msg_client.send_autopilot_msg(ap.Setpoint(chi, ap.Dofs.YAW, True))

        while self.start_event.wait() and not self.stopped_event.isSet() and self.wp_counter < len(self.wp_list):
            pos_msg, wp_list, wp_counter, wp_grad = self.get_info()

            # Get new setpoint
            try:
                chi, delta, e = self.get_los_values(wp_list[wp_counter], wp_list[wp_counter + 1], pos_msg)
            except IndexError:
                logger.info('Final Waypoint reached!')
                break

            # Check if ROA
            if delta < self.roa:
                wp_counter += 1
                self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[wp_counter + 1][2], ap.Dofs.DEPTH, True))
                turn_speed, slow_down_dist = self.turn_vel(wp_counter)
                s = segment_length(wp_list[wp_counter], wp_list[wp_counter])
                if s < LosSettings.roa:
                    self.roa = s
                else:
                    self.roa = LosSettings.roa
                try:
                    chi, delta, e = self.get_los_values(wp_list[wp_counter], wp_list[wp_counter + 1], pos_msg)
                except IndexError:
                    logger.info('Final Waypoint reached!')
                    break
                logger.info('Current wp is WP{}: [N={:.6f}, E={:.6f}, D={:.2f}], ROA={:.2f}'.format(wp_counter,
                                                                                        wp_list[wp_counter][0],
                                                                                        wp_list[wp_counter][1],
                                                                                        wp_list[wp_counter][2],
                                                                                                    self.roa))

            # Check if slow down
            if slow_down_dist >= 0 and delta < slow_down_dist:
                self.set_speed(turn_speed)

            # Send new setpoints
            if abs(chi - self.last_chi) > LosSettings.send_new_heading_limit:
                self.msg_client.send_autopilot_msg(ap.Setpoint(wrapTo2Pi(chi), ap.Dofs.YAW, True))
                self.last_chi = chi

            with self.lock:
                self.wp_counter = wp_counter
                self.e = e
                self.delta = delta

        self.msg_client.stop_autopilot()
        logger.info('WP loop finished')

    def update_pos(self, msg):
        # self.lock.acquire()
        with self.lock:
            self.pos_msg = msg
        # self.lock.release()

    def update_wps(self, wp_list):
        with self.lock:
            self.wp_list, self.wp_grad = path_grad(wp_list)
            self.wp_counter = 0
            self.new_waypoints = True

    def get_wp_counter(self):
        with self.lock:
            return self.wp_counter

    def update_speed(self, speed):
        self.msg_client.send_autopilot_msg(ap.CruiseSpeed(speed))
        self.surge_speed = speed

    def get_errors(self):
        with self.lock:
            return self.e, self.delta
