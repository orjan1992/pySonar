import re
import io
from messages.udpMsg import *
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import threading, socketserver, socket
from settings import ConnectionSettings
import logging
logger = logging.getLogger('UdpClient')


class UdpClient(QObject):
    # TODO: Implement NMEA messages for WP control or LOS control
    signal_new_sonar_msg = pyqtSignal(object, name='new_sonar_msg')
    buffer = io.BytesIO()
    cur_pos_msg = None
    sonar_callback = None
    autopilot_sid = 0

    def __init__(self, sonar_port, pos_port, wp_ip, wp_port):
        super().__init__()
        self.sonar_port = sonar_port
        self.pos_port = pos_port
        self.wp_ip = wp_ip
        self.wp_port = wp_port

        self.buffer_lock = threading.Lock()
        self.sonar_update_thread = None

        self.sonar_server = socketserver.UDPServer(('0.0.0.0', sonar_port), handler_factory(self.parse_sonar_msg))
        self.sonar_thread = threading.Thread(target=self.sonar_server.serve_forever)
        self.sonar_thread.setDaemon(True)

        self.pos_server = socketserver.UDPServer(('0.0.0.0', pos_port), handler_factory(self.parse_pos_msg))
        self.pos_thread = threading.Thread(target=self.pos_server.serve_forever)
        self.pos_thread.setDaemon(True)

        self.autopilot_server = socketserver.UDPServer(('0.0.0.0', wp_port), handler_factory(self.parse_autopilot_msg))
        self.autopilot_thread = threading.Thread(target=self.autopilot_server.serve_forever)
        self.autopilot_thread.setDaemon(True)

        self.autopilot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        self.sonar_thread.start()
        self.pos_thread.start()
        self.autopilot_thread.start()

    def set_sonar_callback(self, fnc):
        self.sonar_callback = fnc

    def parse_sonar_msg(self, data, socket):
        # self.buffer_lock.acquire(blocking=True)
        self.buffer.write(data)
        tmp = self.buffer.getbuffer()
        for i in range(0, len(tmp)):
            if tmp[i] == 0x40:
                try:
                    msg = MtHeadData(tmp[i:len(tmp)])

                    # self.buffer_lock.acquire(blocking=True)
                    # self.sonar_update_thread = threading.Thread(target=self.sonar_callback, args=[msg])
                    # self.sonar_update_thread.start()
                    # self.buffer_lock.release()

                    self.sonar_callback(msg)
                    tmp = None
                    self.buffer = io.BytesIO()
                except CorruptMsgException:
                    logger.error('Corrupt msg')
                    self.buffer = io.BytesIO()
                except OtherMsgTypeException:
                    self.buffer = io.BytesIO()
                    logger.debug('Other sonar msg')
                except UncompleteMsgException:
                    pass
                except Exception as e:
                    # self.buffer_lock.release()
                    raise e
                break
        # self.buffer_lock.release()

    def send_autopilot_msg(self, msg):
        if self.autopilot_sid != 0:
            msg.sid = self.autopilot_sid
        self.autopilot_socket.sendto(msg.compile(), (self.wp_ip, self.wp_port))

    def parse_pos_msg(self, data, socket):
        msg = UdpPosMsg(data)
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

def handler_factory(callback):
    def createHandler(*args, **keys):
        return Handler(callback, *args, **keys)
    return createHandler


if __name__ == '__main__':
    from settings import ConnectionSettings
    client = UdpClient(ConnectionSettings.sonar_port, ConnectionSettings.pos_port, ConnectionSettings.wp_ip, ConnectionSettings.wp_port)




