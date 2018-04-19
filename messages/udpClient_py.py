import io
from messages.udpMsg import *
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import threading, socketserver, socket
from settings import Settings, CollisionSettings
from messages.SeaNet import SeanetDecode
import messages.AutoPilotMsg as ap
import logging
logger = logging.getLogger('UdpClient')


class UdpClient(QObject):
    # TODO: Implement NMEA messages for WP control or LOS control
    signal_new_sonar_msg = pyqtSignal(object, name='new_sonar_msg')
    buffer = io.BytesIO()
    cur_pos_msg = None
    sonar_callback = None
    autopilot_sid = 0
    seanet = SeanetDecode()

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
        self.sonar_thread = threading.Thread(target=self.sonar_server.serve_forever)
        self.sonar_thread.setDaemon(True)

        if Settings.pos_msg_source == 0:
            self.pos_server = socketserver.UDPServer(('0.0.0.0', pos_port), handler_factory(self.parse_pos_msg))
            self.pos_thread = threading.Thread(target=self.pos_server.serve_forever)
            self.pos_thread.setDaemon(True)
        else:
            self.pos_stop_event = threading.Event()
            self.pos_thread = Wathdog(self.pos_stop_event, self.get_pos, Settings.pos_update_speed/1000.0)
            self.pos_thread.setDaemon(True)

        if CollisionSettings.send_new_wps:
            self.autopilot_watchdog_stop_event = threading.Event()
            self.autopilot_watchdog_thread = Wathdog(self.autopilot_watchdog_stop_event, self.ping_autopilot_server,
                                                     ConnectionSettings.autopilot_watchdog_timeout)
            self.autopilot_watchdog_thread.setDaemon(True)

        if autopilot_listen_port is not None:
            self.autopilot_server = socketserver.UDPServer(('0.0.0.0', autopilot_listen_port), handler_factory(self.parse_autopilot_msg))
            self.autopilot_thread = threading.Thread(target=self.autopilot_server.serve_forever)
            self.autopilot_thread.setDaemon(True)

    def start(self):
        self.sonar_thread.start()
        self.pos_thread.start()
        if self.autopilot_listen_port is not None:
            self.autopilot_thread.start()
        if CollisionSettings.send_new_wps:
            self.autopilot_watchdog_thread.start()

    def close(self):
        self.autopilot_watchdog_stop_event.set()
        self.send_autopilot_msg(ap.RemoteControlRequest(True))

    def set_sonar_callback(self, fnc):
        self.sonar_callback = fnc

    def parse_sonar_msg(self, data):
        try:
            data_packet = self.seanet.add(data)
            if data_packet is not None:
                msg = MtHeadData(data_packet)
                self.sonar_callback(msg)
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
            self.send_autopilot_msg(ap.RemoteControlRequest(True))
            logger.info("Asked for remote control")
        else:
            if Settings.pos_msg_source == 1:
                self.autopilot_watchdog_stop_event.set()
            self.send_autopilot_msg(ap.GetMessage(ap.MsgType.EMPTY_MESSAGE))

    def get_pos(self):
        self.send_autopilot_msg(ap.GetMessage(ap.MsgType.ROV_STATE))

    def update_wps(self, wp_list):
        self.send_autopilot_msg(ap.TrackingSpeed(0))
        self.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.STATION_KEEPING))
        self.send_autopilot_msg(ap.Command(ap.CommandOptions.CLEAR_WPS))
        self.send_autopilot_msg(ap.AddWaypoints(wp_list))
        self.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.PATH_FOLLOWING))
        if len(wp_list[0]) > 3:
            self.send_autopilot_msg(ap.TrackingSpeed(wp_list[0][3]))
        else:
            self.send_autopilot_msg(ap.TrackingSpeed(CollisionSettings.tracking_speed))

    def parse_pos_msg(self, data):
        msg = UdpPosMsg(data)
        # print(msg)
        if not msg.error:
            self.cur_pos_msg = msg

    def parse_autopilot_msg(self, data):
        try:
            msg = ap.Binary.parse(data)
            if msg.msg_id is ap.MsgType.ROV_STATE:
                self.cur_pos_msg = msg
            elif msg.msg_id is ap.MsgType.REMOTE_CONTROL_REQUEST_REPLY:
                if msg.acquired:
                    self.autopilot_sid = msg.token
                    logger.info('Received remote control token: {}'.format(msg.token))
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
    def __init__(self, event, fnc, timeout):
        super().__init__()
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
    from settings import ConnectionSettings
    client = UdpClient(ConnectionSettings.sonar_port, ConnectionSettings.pos_port, ConnectionSettings.autopilot_ip, ConnectionSettings.autopilot_server_port)




