from settings import CollisionSettings, LosSettings
import threading
import messages.AutoPilotMsg as ap
import logging
import numpy as np
from coordinate_transformations import wrapToPi, wrapTo2Pi
logger = logging.getLogger('LosController')

class LosController:
    wp_list = []
    wp_counter = 0
    pos_msg = None
    last_chi = 0
    counter = 0
    e = 0
    s = 0

    def __init__(self, msg_client, timeout, stopped_event=threading.Event(), start_event=threading.Event()):
        self.normal_surge_speed = CollisionSettings.tracking_speed
        self.surge_speed = self.normal_surge_speed
        self.msg_client = msg_client
        self.timeout = timeout
        self.stopped_event = stopped_event
        self.start_event = start_event
        self.lock = threading.Lock()
        self.new_waypoints = True

    def loop(self):
        if len(self.wp_list) < 2 or self.pos_msg is None:
            self.start_event.wait()
        self.msg_client.switch_ap_mode(ap.GuidanceModeOptions.CRUISE_MODE)
        logger.info('Waiting on correct heading')
        while self.start_event.wait() and not self.stopped_event.isSet() and self.wp_counter < len(self.wp_list):
            self.start_event.clear()
            with self.lock:
                pos_msg = self.pos_msg
                wp_list = self.wp_list
                wp_counter = self.wp_counter
                new_wps = self.new_waypoints
            try:
                alpha = np.arctan2((wp_list[wp_counter+1][1] -
                                    wp_list[wp_counter][1]),
                                   (wp_list[wp_counter+1][0] -
                                    wp_list[wp_counter][0]))
            except IndexError:
                break
            e = -(wp_list[wp_counter+1][0] - pos_msg.lat) * np.sin(alpha) + \
                (wp_list[wp_counter+1][1] - pos_msg.long) * np.cos(alpha)
            # dist = ((wp_list[wp_counter][0] - pos_msg.lat) ** 2 +
            #         (wp_list[wp_counter][1] - pos_msg.long) ** 2)**0.5
            s = ( wp_list[wp_counter+1][0] - pos_msg.lat)*np.cos(alpha) + \
                (wp_list[wp_counter+1][1] - pos_msg.long)*np.cos(alpha)
            # logger.debug(
            #     'waypoint: {:2d}\talpha: {:5.2f}\tCrossTrack: {:5.2f}'.format(wp_counter, alpha * 180 / pi, e))
            chi_r = np.arctan(-e / LosSettings.look_ahead_distance)
            chi = chi_r + alpha
            if new_wps:
                if abs(wrapToPi(chi - pos_msg.psi)) < 4*np.pi/180.0:
                    new_wps = False
                    self.update_speed(self.normal_surge_speed)
                    logger.info('Heading error is small, adjusting surge speed to: {}'.format(self.normal_surge_speed))
                else:
                    if self.counter % 10 == 0:
                        logger.info('Adjusting heading to: {:.2f}'.format(wrapTo2Pi(chi)*180.0/np.pi))
                    self.counter += 1

            # adjust speed if turn is to sharp
            if self.surge_speed != LosSettings.safe_turning_speed:
                try:
                    next_alpha = np.arctan2((wp_list[wp_counter + 2][1] -
                                        wp_list[wp_counter+1][1]),
                                       (wp_list[wp_counter + 2][0] -
                                        wp_list[wp_counter+1][0]))
                    if abs(wrapToPi(self.pos_msg.psi - next_alpha)) > LosSettings.max_heading_change and s < LosSettings.braking_distance:
                        self.update_speed(LosSettings.safe_turning_speed)
                        logger.info('Braking because of sharp turn')
                except IndexError:
                    pass

            # brake if close to final wp
            if wp_counter + 1 >= len(wp_list) and s > LosSettings.braking_distance:
                logger.info('Braking because of last wp')
                self.update_speed(0)

            # Adjust back again
            if self.surge_speed < self.normal_surge_speed and s > LosSettings.braking_distance:
                self.update_speed(self.normal_surge_speed)
                logger.info('Back to normal speed')

            if s < LosSettings.roa:
                if wp_counter + 1 >= len(wp_list):
                    logger.info('Final WP reached')
                    break
                wp_counter += 1
                logger.info('WP counter: {}'.format(wp_counter))
            # print(chi*180.0/np.pi)
            if abs(chi - self.last_chi) > LosSettings.send_new_heading_limit:
                self.msg_client.send_autopilot_msg(ap.Setpoint(wrapTo2Pi(chi), ap.Dofs.YAW, True))
            self.last_chi = chi
            # TODO: Check curvature and reduce speed
            # self.lock.release()
            with self.lock:
                self.wp_counter = wp_counter
                self.e = e
                self.s = s
                self.new_waypoints = new_wps
        self.msg_client.stop_autopilot()
        logger.info('WP loop finished')

    def update_pos(self, msg):
        # self.lock.acquire()
        with self.lock:
            self.pos_msg = msg
        # self.lock.release()

    def update_wps(self, wp_list):
        with self.lock:
            self.wp_list = wp_list
            self.wp_counter = 0
            self.new_waypoints = True

    def update_speed(self, speed):
        self.msg_client.send_autopilot_msg(ap.CruiseSpeed(speed))
        self.surge_speed = speed

    def get_errors(self):
        with self.lock:
            return self.e, self.s
