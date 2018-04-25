from settings import CollisionSettings, LosSettings
import threading
import messages.AutoPilotMsg as ap
import logging
import numpy as np
from coordinate_transformations import wrapToPiHalf
logger = logging.getLogger('LosController')


class LosController(threading.Thread):
    wp_list = []
    wp_counter = 0
    pos_msg = None
    last_chi = 0
    counter = 0

    def __init__(self, msg_client, timeout, stopped_event=threading.Event(), **kwargs):
        super().__init__(**kwargs, daemon=True)
        self.surge_speed = CollisionSettings.tracking_speed
        self.msg_client = msg_client
        self.timeout = timeout
        self.stopped_event = stopped_event
        self.lock = threading.Lock()

    def run(self):
        self.msg_client.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.CRUISE_MODE))
        first = True
        logger.info('Waiting on correct heading')
        while not self.stopped_event.wait(self.timeout) and self.wp_counter < len(self.wp_list):
            # self.lock.acquire()
            alpha = np.arctan2((self.wp_list[self.wp_counter+1][1] -
                                self.wp_list[self.wp_counter][1]),
                               (self.wp_list[self.wp_counter+1][0] -
                                self.wp_list[self.wp_counter][0]))
            e = -(self.pos_msg.lat - self.wp_list[self.wp_counter+1][0]) * np.sin(alpha) + \
                (self.pos_msg.long - self.wp_list[self.wp_counter+1][1]) * np.cos(alpha)
            # dist = ((self.wp_list[self.wp_counter][0] - self.pos_msg.lat) ** 2 +
            #         (self.wp_list[self.wp_counter][1] - self.pos_msg.long) ** 2)**0.5
            s = ( self.wp_list[self.wp_counter+1][0] - self.pos_msg.lat)*np.cos(alpha) + \
                (self.wp_list[self.wp_counter+1][1] - self.pos_msg.long)*np.cos(alpha)
            # logger.debug(
            #     'waypoint: {:2d}\talpha: {:5.2f}\tCrossTrack: {:5.2f}'.format(self.wp_counter, alpha * 180 / pi, e))
            chi_r = np.arctan(-e / LosSettings.look_ahead_distance)
            chi = chi_r + alpha
            if first:
                if abs(wrapToPiHalf(chi - self.pos_msg.psi)) < 4*np.pi/180.0:
                    first = False
                    self.msg_client.send_autopilot_msg(ap.CruiseSpeed(self.surge_speed))
                    logger.info('Heading error is small, adjusting surge speed')
                else:
                    logger.info('Adjusting heading. Current heading: {}'.format(self.pos_msg.psi))


            if s < LosSettings.roa:
                if self.wp_counter + 1 >= len(self.wp_list):
                    logger.info('Final WP reached')
                    break
                self.wp_counter += 1
                logger.info('WP counter: {}'.format(self.wp_counter))
            # print(chi*180.0/np.pi)
            if abs(chi - self.last_chi) > LosSettings.send_new_heading_limit:
                self.msg_client.send_autopilot_msg(ap.Setpoint(chi, ap.Dofs.YAW, True))
            self.last_chi = chi
            # TODO: Check curvature and reduce speed
            # self.lock.release()
        logger.info('WP loop finished')
        self.msg_client.send_autopilot_msg(ap.CruiseSpeed(0))

    def update_pos(self, msg):
        # self.lock.acquire()
        self.pos_msg = msg
        # self.lock.release()

    def update_wps(self, wp_list):
        self.lock.acquire()
        self.wp_list = wp_list
        self.wp_counter = 0
        self.lock.release()