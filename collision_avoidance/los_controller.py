from settings import CollisionSettings, LosSettings, Settings
import threading
import messages.AutoPilotMsg as ap
import logging
import numpy as np
from coordinate_transformations import wrapToPi, wrapTo2Pi
from collision_avoidance.path_smoothing import path_grad
from datetime import datetime
from scipy.io import savemat
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
    segment_lengths = []
    wp_counter = 0
    pos_msg = None
    last_chi = 0
    counter = 0
    e = 0
    delta = 0
    turn_velocity = 0
    roa = LosSettings.roa

    def __init__(self, msg_client, timeout, stopped_event=threading.Event(), start_event=threading.Event(),
                 restart_event=threading.Event()):
        self.normal_surge_speed = LosSettings.cruise_speed
        self.surge_speed = 0
        self.msg_client = msg_client
        self.timeout = timeout
        self.stopped_event = stopped_event
        self.start_event = start_event
        self.restart_event = restart_event
        self.lock = threading.Lock()
        self.new_waypoints = True

    def get_info(self):
        with self.lock:
            self.start_event.clear()
            return self.pos_msg,  self.wp_list, self.wp_counter, self.wp_grad, self.segment_lengths

    def get_pos(self):
        with self.lock:
            self.start_event.clear()
            return self.pos_msg

    def get_los_values(self, wp1, wp2, pos):
        alpha = get_angle(wp1, wp2)
        e, s = get_errors(wp1, (pos.north, pos.east), alpha)
        delta = segment_length(wp1, wp2) - s
        if self.surge_speed == 0:
            chi_r = np.arctan(-e / (self.normal_surge_speed*LosSettings.look_ahead_time))
        else:
            chi_r = np.arctan(-e / (self.surge_speed*LosSettings.look_ahead_time))
        # print(chi_r*180.0/np.pi)
        chi = chi_r + alpha
        return wrapToPi(chi), delta, e

    def turn_vel(self, i):
        # if self.wp_grad[i] == 0 and len(self.wp_list) > i + 2:
        #     logger.info('turn_vel: {:.2f}, turn dist: {:.2f}'.format(self.turn_velocity, -1))
        #     self.surge_speed = self.normal_surge_speed
        #     return self.normal_surge_speed, -1
        if len(self.wp_list) <= i + 2:
            self.turn_velocity = 0
        else:
            self.turn_velocity = min(self.normal_surge_speed, LosSettings.safe_turning_speed / (8 * self.wp_grad[i]))
            # TODO: Something to find further gradient as well
            if self.segment_lengths[i] < LosSettings.roa:
                try:
                    self.turn_velocity = min(self.turn_velocity,
                                             LosSettings.safe_turning_speed / (8 * self.wp_grad[i+1]))
                except IndexError:
                    pass

        diff = self.normal_surge_speed - self.turn_velocity
        dist = -1
        if diff > 0:
            t = diff / LosSettings.max_acc
            dist = self.surge_speed*t + 0.5*LosSettings.max_acc*t**2
        logger.info('turn_vel: {:.2f}, turn dist: {:.2f}'.format(self.turn_velocity, dist))
        return self.turn_velocity, dist

    def set_speed(self, speed):
        if speed != self.surge_speed:
            self.surge_speed = speed
            if Settings.input_source == 0:
                self.msg_client.send_autopilot_msg(ap.CruiseSpeed(speed))
            else:
                self.msg_client.send_msg('vel_com', speed)
            logger.info('Surge speed adjusted to {:.2f} m/s'.format(speed))

    def loop(self):
        logger.info('New path received')
        if len(self.wp_list) < 2 or self.pos_msg is None:
            logger.info('To few waypoints. Returning to Stationkeeping.')
            self.msg_client.stop_autopilot()
            return

        if LosSettings.log_paths:
            chi_log = []
            surge_log = []
            timestamp = []
            delta_log = []
            e_log = []
            wp_change = []
            roa_log = []
            slow_down_dist_log = []
            slown_down_vel_log = []
            pos_log = []

        # self.surge_speed = 0
        # Initial check
        pos_msg, wp_list, wp_counter, wp_grad, segment_lengths = self.get_info()
        chi, delta, e = self.get_los_values(wp_list[0], wp_list[1], pos_msg)
        start_chi = wrapToPi(get_angle(wp_list[0], wp_list[1]))
        logger.info('Start Chi: {:.2f}'.format(start_chi*180.0/np.pi))
        # TODO: CHeck if on initial path
        if (segment_length(wp_list[0], (pos_msg.north, pos_msg.east)) > LosSettings.roa or
                abs(start_chi - pos_msg.yaw) > LosSettings.start_heading_diff) and \
                not (segment_lengths[0] - delta > 0 and abs(e) < LosSettings.roa):
            logger.info('Not on path, s: {:.2f}, e: {:.2f}, d_psi: {:.2f}'.format(segment_lengths[0] - delta, e, (start_chi - pos_msg.yaw)*180.0/np.pi))
            # Wait for low speed before stationkeeping
            # logger.info('Speed to high to start LOS-guidance')
            # self.set_speed(0)
            # while self.start_event.wait() and not self.stopped_event.isSet():
            #     pos_msg = self.get_pos()
            #     if pos_msg.v_surge < 0.01:
            #         break
            th = self.msg_client.stop_autopilot()
            th.join()
            # self.msg_client.switch_ap_mode(ap.GuidanceModeOptions.STATION_KEEPING)
            if Settings.input_source == 0:
                self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][0], ap.Dofs.NORTH, True))
                self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][1], ap.Dofs.EAST, True))
                self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][2], ap.Dofs.DEPTH, True))
                self.msg_client.send_autopilot_msg(ap.Setpoint(start_chi, ap.Dofs.YAW, True))
            else:
                self.msg_client.send_msg('north_com', wp_list[0][0])
                self.msg_client.send_msg('east_com', wp_list[0][1])
                self.msg_client.send_msg('depth_com', wp_list[0][2])
                self.msg_client.send_msg('yaw_com', start_chi)
            # logger.info('To far from inital setpoint. Moving to: '
            #             '[N={:.6f}, E={:.6f}, D={:.2f}, YAW={:.2f} deg]'.format(wp_list[0][0], wp_list[0][1],
            #                                                       wp_list[0][2], start_chi*180.0/np.pi))
            logger.info('To far from inital setpoint. Yaw_hat:{:.2f}, north_hat: {:.2f}, east_hat: {:.2f}'.format(
                (start_chi - pos_msg.yaw)*180.0/np.pi, wp_list[0][0]-pos_msg.north, wp_list[0][1]-pos_msg.east
            ))

            while self.start_event.wait() and not self.stopped_event.isSet() and \
                    (segment_length(wp_list[0], (pos_msg.north, pos_msg.east)) > 1 or
                            abs(start_chi - pos_msg.yaw) > LosSettings.start_heading_diff):
                pos_msg = self.get_pos()

        if Settings.input_source == 0:
            self.msg_client.switch_ap_mode(ap.GuidanceModeOptions.CRUISE_MODE)
        # self.set_speed(self.normal_surge_speed)
        turn_speed, slow_down_dist = self.turn_vel(0)
        if segment_lengths[wp_counter] < LosSettings.roa:
            self.roa = segment_lengths[wp_counter] / 2
        else:
            self.roa = LosSettings.roa
        if LosSettings.log_paths:
            roa_log.append(self.roa)
            slow_down_dist_log.append(slow_down_dist)
            slown_down_vel_log.append(turn_speed)
        # logger.info('Setting cruise speed: {} m/s'.format(self.surge_speed))
        if Settings.input_source == 0:
            self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[0][2], ap.Dofs.DEPTH, True))
            self.msg_client.send_autopilot_msg(ap.Setpoint(chi, ap.Dofs.YAW, True))
        else:
            self.msg_client.send_msg('depth_com', wp_list[0][2])
            self.msg_client.send_msg('yaw_com', wrapToPi(chi))
        if (slow_down_dist < 0) or (delta <= slow_down_dist):
            self.set_speed(turn_speed)
        else:
            self.set_speed(self.normal_surge_speed)
        if LosSettings.log_paths:
            chi_log.append(chi)
            surge_log.append(self.surge_speed)
            timestamp.append(datetime.now().strftime("%H:%M:%S"))
            delta_log .append(delta)
            e_log.append(e)
            wp_change.append(datetime.now().strftime("%H:%M:%S"))
            pos_log.append(pos_msg.to_tuple())
        logger.info('Path Started')
        while not self.stopped_event.isSet() and self.start_event.wait() and self.wp_counter < len(self.wp_list):
            pos_msg = self.get_pos()

            # Get new setpoint
            chi, delta, e = self.get_los_values(wp_list[wp_counter], wp_list[wp_counter + 1], pos_msg)

            # # Check if slow down
            # if slow_down_dist >= 0 and delta <= slow_down_dist:
            #     self.set_speed(turn_speed)

            # Check if ROA
            if delta < self.roa:
                wp_counter += 1
                if LosSettings.log_paths:
                    wp_change.append(datetime.now().strftime("%H:%M:%S"))
                try:
                    turn_speed, slow_down_dist = self.turn_vel(wp_counter)
                    if segment_lengths[wp_counter] < LosSettings.roa:
                        self.roa = segment_lengths[wp_counter] / 2
                    else:
                        self.roa = LosSettings.roa
                    if LosSettings.log_paths:
                        roa_log.append(self.roa)
                        slow_down_dist_log.append(slow_down_dist)
                        slown_down_vel_log.append(turn_speed)
                    if Settings.input_source == 0:
                        self.msg_client.send_autopilot_msg(ap.Setpoint(wp_list[wp_counter + 1][2], ap.Dofs.DEPTH, True))
                    else:
                        self.msg_client.send_msg('alt_com', 2.0)
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
            if (slow_down_dist < 0) or (delta <= slow_down_dist):
                self.set_speed(turn_speed)
            else:
                self.set_speed(self.normal_surge_speed)

            # Send new setpoints
            if abs(chi - self.last_chi) > LosSettings.send_new_heading_limit:
                # logger.info('Chi: {:2f}'.format(wrapToPi(chi)*180.0/np.pi))
                if Settings.input_source == 0:
                    self.msg_client.send_autopilot_msg(ap.Setpoint(wrapToPi(chi), ap.Dofs.YAW, True))
                else:
                    self.msg_client.send_msg('yaw_com', wrapToPi(chi))
                self.last_chi = chi
            if LosSettings.log_paths:
                chi_log.append(chi)
                surge_log.append(self.surge_speed)
                timestamp.append(datetime.now().strftime("%H:%M:%S"))
                delta_log .append(delta)
                e_log.append(e)
                pos_log.append(pos_msg.to_tuple())
            with self.lock:
                self.wp_counter = wp_counter
                self.e = e
                self.delta = delta
        with self.lock:
            self.e = 0
            self.delta = 0
        if not self.restart_event.isSet():
            logger.info('WP loop finished: Stopping ROV')
            self.msg_client.stop_autopilot()
        else:
            logger.info('WP loop finished: Restarting loop')
        if LosSettings.log_paths:
            savemat('{}Los_log_{}'.format(Settings.log_folder, datetime.now().strftime("%Y%m%d-%H%M%S")), mdict={
                'chi': np.array(chi_log), 'surge': np.array(surge_log), 'time': timestamp, 'path': wp_list,
                'delta': np.array(delta_log), 'cross_track': np.array(e_log),
                'slow_down_dist': np.array(slow_down_dist_log), 'roa': np.array(roa_log),
                'slow_down_vel': np.array(slown_down_vel_log), 'wp_change': wp_change, 'pos': np.array(pos_log)})


    def update_pos(self, msg):
        # self.lock.acquire()
        with self.lock:
            self.pos_msg = msg
        # self.lock.release()

    def update_wps(self, wp_list):
        with self.lock:
            logger.info('Updated waypoints')
            self.wp_list, self.wp_grad, self.segment_lengths = path_grad(wp_list)
            self.wp_counter = 0
            self.new_waypoints = True

    def get_wp_counter(self):
        with self.lock:
            return self.wp_counter

    def get_errors(self):
        with self.lock:
            return self.e, self.delta
