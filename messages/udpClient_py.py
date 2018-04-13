import io
from messages.udpMsg import *
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import threading, socketserver, socket
from settings import Settings, CollisionSettings
from messages.SeaNet import SeanetDecode
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

    def __init__(self, sonar_port, pos_port, autopilot_ip, autopilot_port):
        super().__init__()
        self.sonar_port = sonar_port
        self.pos_port = pos_port
        self.autopilot_ip = autopilot_ip
        self.autopilot_port = autopilot_port

        self.buffer_lock = threading.Lock()
        self.sonar_update_thread = None

        self.sonar_server = socketserver.UDPServer(('0.0.0.0', sonar_port), handler_factory(self.parse_sonar_msg))
        self.sonar_server.allow_reuse_address = True
        self.sonar_thread = threading.Thread(target=self.sonar_server.serve_forever)
        self.sonar_thread.setDaemon(True)

        self.pos_server = socketserver.UDPServer(('0.0.0.0', pos_port), handler_factory(self.parse_pos_msg))
        self.pos_thread = threading.Thread(target=self.pos_server.serve_forever)
        self.pos_thread.setDaemon(True)

        if autopilot_port is not None:
            self.autopilot_server = socketserver.UDPServer(('0.0.0.0', autopilot_port), handler_factory(self.parse_autopilot_msg))
            self.autopilot_thread = threading.Thread(target=self.autopilot_server.serve_forever)
            self.autopilot_thread.setDaemon(True)

        if CollisionSettings.send_new_wps:
            self.autopilot_watchdog_stop_event = threading.Event()
            self.autopilot_watchdog_thread = Wathdog(self.autopilot_watchdog_stop_event, self.ping_autopilot_server,
                                                     ConnectionSettings.autopilot_watchdog_timeout)
            self.autopilot_watchdog_thread.setDaemon(True)

        self.autopilot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        self.sonar_thread.start()
        self.pos_thread.start()
        if self.autopilot_port is not None:
            self.autopilot_thread.start()
        if CollisionSettings.send_new_wps:
            self.autopilot_watchdog_thread.start()

    def set_sonar_callback(self, fnc):
        self.sonar_callback = fnc

    def parse_sonar_msg(self, data, socket):
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
        self.autopilot_socket.sendto(msg.compile(), (self.autopilot_ip, self.autopilot_port))

    def ping_autopilot_server(self):
        # Get empty msg to keep control
        self.send_autopilot_msg(AutoPilotGetMessage(19))

    def parse_pos_msg(self, data, socket):
        msg = UdpPosMsg(data)
        # print(msg)
        if not msg.error:
            self.cur_pos_msg = msg

    def parse_autopilot_msg(self, data, socket):
        try:
            msg = AutoPilotBinary.parse(data)
            if msg is AutoPilotRemoteControlRequestReply:
                if msg.acquired:
                    self.autopilot_sid = msg.token
            else:
                raise NotImplemented
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
        socket = self.request[1]
        self.callback(data, socket)

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
    client = UdpClient(ConnectionSettings.sonar_port, ConnectionSettings.pos_port, ConnectionSettings.autopilot_ip, ConnectionSettings.autopilot_port)




