import io
from messages.udpMsg import *
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import threading, socketserver, socket
from settings import Settings, CollisionSettings, LosSettings
from collision_avoidance.los_controller import LosController
from messages.SeaNet import SeanetDecode
import messages.AutoPilotMsg as ap
import logging
logger = logging.getLogger('UdpClient')

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class UdpClient(QObject):
    # TODO: Implement NMEA messages for WP control or LOS control
    signal_new_sonar_msg = pyqtSignal(object, name='new_sonar_msg')
    buffer = io.BytesIO()
    cur_pos_msg = None
    cur_desired_pos_msg = None
    sonar_callback = None
    autopilot_sid = 0
    seanet = SeanetDecode()
    ask_for_desired = False
    in_control = False
    wp_update_in_progress = threading.Lock()

    def __init__(self, sonar_port, pos_port, autopilot_ip, autopilot_server_port, autopilot_listen_port):
        super().__init__()
        self.sonar_port = sonar_port
        self.pos_port = pos_port
        self.autopilot_ip = autopilot_ip
        self.autopilot_server_port = autopilot_server_port
        self.autopilot_listen_port = autopilot_listen_port

        self.buffer_lock = threading.Lock()
        self.sonar_update_thread = None

        self.sonar_server = socketserver.UDPServer(('0.0.0.0', sonar_port), handler_factory(self.parse_sonar_msg))
        self.sonar_server.allow_reuse_address = True
        self.sonar_thread = threading.Thread(target=self.sonar_server.serve_forever, daemon=True)

        if Settings.pos_msg_source == 0:
            self.pos_server = socketserver.UDPServer(('0.0.0.0', pos_port), handler_factory(self.parse_pos_msg))
            self.pos_thread = threading.Thread(target=self.pos_server.serve_forever, daemon=True)
        else:
            self.pos_stop_event = threading.Event()
            self.pos_thread = Wathdog(self.pos_stop_event, self.get_pos, Settings.pos_update_speed/1000.0, daemon=True)

        if CollisionSettings.send_new_wps:
            self.autopilot_watchdog_stop_event = threading.Event()
            self.autopilot_watchdog_thread = Wathdog(self.autopilot_watchdog_stop_event, self.ping_autopilot_server,
                                                     ConnectionSettings.autopilot_watchdog_timeout, daemon=True)
            self.ap_pos_received = threading.Event()
            self.guidance_mode = ap.GuidanceModeOptions.STATION_KEEPING

        if autopilot_listen_port is not None:
            self.autopilot_server = socketserver.UDPServer(('0.0.0.0', autopilot_listen_port),
                                                           handler_factory(self.parse_autopilot_msg))
            self.autopilot_thread = threading.Thread(target=self.autopilot_server.serve_forever, daemon=True)
        if LosSettings.enable_los:
            self.los_stop_event = threading.Event()
            self.los_controller = LosController(self, 0.05, self.los_stop_event)

    def start(self):
        self.sonar_thread.start()
        self.pos_thread.start()
        if self.autopilot_listen_port is not None:
            self.autopilot_thread.start()
        if CollisionSettings.send_new_wps and self.autopilot_server_port is not None:
            self.autopilot_watchdog_thread.start()
        # if LosSettings.enable_los:
        #     self.los_controller.start()

    def close(self):
        if LosSettings.enable_los:
            self.los_stop_event.set()
        if CollisionSettings.send_new_wps and self.in_control:
            logger.info('Shutting down autopilot')
            self.ask_for_desired = True
            thread = self.stop_autopilot()
            thread.join()
            logger.info('In stationkeeping mode. Waiting for small errors.')
            state_error = None
            self.ap_pos_received.clear()
            while state_error is None or not state_error.is_small():
                self.ap_pos_received.wait()
                if self.cur_desired_pos_msg is None or self.cur_pos_msg is None:
                    self.ap_pos_received.clear()
                    continue
                state_error = self.cur_pos_msg - self.cur_desired_pos_msg
            # All errors are small
            self.send_autopilot_msg(ap.RemoteControlRequest(False))
            logger.info('Released control of autopilot')
        if CollisionSettings.send_new_wps:
            self.autopilot_watchdog_stop_event.set()

    def set_sonar_callback(self, fnc):
        self.sonar_callback = fnc

    def parse_sonar_msg(self, data):
        try:
            data_packet = self.seanet.add(data)
            if data_packet is not None:
                msg = MtHeadData(data_packet)
                # self.sonar_callback(msg)
                self.signal_new_sonar_msg.emit(msg)
        except CorruptMsgException:
            logger.error('Corrupt msg')
        except OtherMsgTypeException:
            logger.debug('Other sonar msg')
        except UncompleteMsgException:
            logger.debug('Uncomplete sonar msg')

    def send_autopilot_msg(self, msg):
        if self.autopilot_sid != 0:
            msg.sid = self.autopilot_sid
        self.autopilot_server.socket.sendto(msg.compile(), (self.autopilot_ip, self.autopilot_server_port))

    def ping_autopilot_server(self):
        # Get empty msg to keep control
        if self.autopilot_sid == 0:
            # self.send_autopilot_msg(ap.ControllerOptions([]))
            self.send_autopilot_msg(ap.RemoteControlRequest(True))
            logger.info("Asked for remote control")
            return False
        else:
            self.in_control = True
            if Settings.pos_msg_source == 1:
                # self.send_autopilot_msg(ap.ControllerOptions([ap.Dofs.SURGE, ap.Dofs.SWAY, ap.Dofs.HEAVE, ap.Dofs.YAW]))
                self.autopilot_watchdog_stop_event.set()
                self.guidance_mode = ap.GuidanceModeOptions.STATION_KEEPING
                self.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.STATION_KEEPING))
                # while self.cur_pos_msg is None:
                #     pass
                # self.send_autopilot_msg(ap.Setpoint(self.cur_pos_msg.lat, ap.Dofs.SURGE, True))
                # self.send_autopilot_msg(ap.Setpoint(self.cur_pos_msg.long, ap.Dofs.SWAY, True))
                # self.send_autopilot_msg(ap.Setpoint(self.cur_pos_msg.psi, ap.Dofs.YAW, True))
                # self.send_autopilot_msg(ap.VerticalPos(ap.VerticalPosOptions.ALTITUDE, 10))
            else:
                self.send_autopilot_msg(ap.GetMessage(ap.MsgType.EMPTY_MESSAGE))
            return True

    def get_pos(self):
        if not self.ask_for_desired:
            self.send_autopilot_msg(ap.GetMessage(ap.MsgType.ROV_STATE))
        else:
            self.send_autopilot_msg(ap.GetMessage([ap.MsgType.ROV_STATE, ap.MsgType.ROV_STATE_DESIRED]))

    @threaded
    def update_wps(self, wp_list):
        if len(wp_list) < 2:
            return
        if LosSettings.enable_los:
            self.los_stop_event = threading.Event()
            self.los_controller = LosController(self, 0.05, self.los_stop_event)
            self.los_controller.update_pos(self.cur_pos_msg)
            self.los_controller.wp_list = wp_list
            self.los_controller.start()
            return
        self.wp_update_in_progress.acquire()
        thread = self.stop_autopilot()
        thread.join()
        self.send_autopilot_msg(ap.Command(ap.CommandOptions.CLEAR_WPS))
        self.send_autopilot_msg(ap.AddWaypoints(wp_list))
        self.guidance_mode = ap.GuidanceModeOptions.PATH_FOLLOWING
        self.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.PATH_FOLLOWING))
        # self.send_autopilot_msg(ap.Setpoint(0, ap.Dofs.YAW, False))
        # if len(wp_list[0]) > 3:
        #     self.send_autopilot_msg(ap.TrackingSpeed(wp_list[0][3]))
        # else:
        self.send_autopilot_msg(ap.TrackingSpeed(CollisionSettings.tracking_speed))
        self.wp_update_in_progress.release()

    @threaded
    def stop_autopilot(self):
        if self.cur_pos_msg is None:
            return
        if not self.guidance_mode is ap.GuidanceModeOptions.STATION_KEEPING:
            if self.guidance_mode is ap.GuidanceModeOptions.PATH_FOLLOWING:
            # self.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.PATH_FOLLOWING))
                self.send_autopilot_msg(ap.TrackingSpeed(0))
            if self.guidance_mode is ap.GuidanceModeOptions.CRUISE_MODE:
                    self.send_autopilot_msg(ap.CruiseSpeed(0))
            while self.cur_pos_msg.surge > 0.1:
                self.ap_pos_received.clear()
                self.ap_pos_received.wait(0.1)
            self.guidance_mode = ap.GuidanceModeOptions.STATION_KEEPING
            self.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.STATION_KEEPING))
            # self.send_autopilot_msg(ap.Setpoint(0, ap.Dofs.SURGE, True))
            # self.send_autopilot_msg(ap.Setpoint(0, ap.Dofs.SWAY, True))
            # self.send_autopilot_msg(ap.Setpoint(self.cur_pos_msg.psi, ap.Dofs.YAW, True))
            self.send_autopilot_msg(ap.VerticalPos(ap.VerticalPosOptions.ALTITUDE, self.cur_pos_msg.alt))
        else:
            return

    def parse_pos_msg(self, data):
        msg = UdpPosMsg(data)
        # print(msg)
        if not msg.error:
            self.cur_pos_msg = msg
            if LosSettings.enable_los:
                self.los_controller.update_pos(msg)

    def parse_autopilot_msg(self, data):
        try:
            msg = ap.Binary.parse(data)
            if msg.msg_id is ap.MsgType.ROV_STATE:
                self.cur_pos_msg = msg
                self.ap_pos_received.set()
            elif msg.msg_id is ap.MsgType.REMOTE_CONTROL_REQUEST_REPLY:
                if msg.acquired:
                    self.autopilot_sid = msg.token
                    logger.info('Received remote control token: {}'.format(msg.token))
            elif msg.msg_id is ap.MsgType.ROV_STATE_DESIRED:
                self.cur_desired_pos_msg = msg
                self.ap_pos_received.set()
            else:
                raise OtherMsgTypeException
        except OtherMsgTypeException:
            logger.debug('Unknown msg from autopilot server')
        except CorruptMsgException:
            logger.debug('Corrupt msg from autopilot server')

class Handler(socketserver.BaseRequestHandler):
    """ One instance per connection. """
    def __init__(self, callback, *args, **keys):
        self.callback = callback
        socketserver.BaseRequestHandler.__init__(self, *args, **keys)

    def handle(self):
        data = self.request[0].strip()
        # socket = self.request[1]
        self.callback(data)

class Wathdog(threading.Thread):
    def __init__(self, event, fnc, timeout, **kwargs):
        super().__init__(**kwargs)
        self.stopped = event
        self.fnc = fnc
        self.timeout = timeout

    def run(self):
        while not self.stopped.wait(self.timeout):
            self.fnc()

def handler_factory(callback):
    def createHandler(*args, **keys):
        return Handler(callback, *args, **keys)
    return createHandler


if __name__ == '__main__':
    from time import sleep
    from settings import ConnectionSettings
    client = UdpClient(ConnectionSettings.sonar_port, ConnectionSettings.pos_port, ConnectionSettings.autopilot_ip, ConnectionSettings.autopilot_server_port, ConnectionSettings.autopilot_listen_port)
    client.start()
    while not client.in_control:
        pass
    client.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.CRUISE_MODE))
    client.send_autopilot_msg(ap.CruiseSpeed(0.7))
    counter = 0
    while counter < 4:
        sleep(5)
        client.send_autopilot_msg(ap.Setpoint(counter*np.pi/4, ap.Dofs.YAW, True))
        counter += 1
    client.send_autopilot_msg(ap.CruiseSpeed(0))
    print('closing down')
    client.close()
    print('closed')




