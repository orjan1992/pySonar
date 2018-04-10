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

    def start(self):
        self.sonar_thread.start()
        self.pos_thread.start()

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

                    self.buffer = io.BytesIO()
                except CorruptMsgException:
                    self.buffer = io.BytesIO()
                except OtherMsgTypeException:
                    self.buffer = io.BytesIO()
                    print("Other msg type")
                except UncompleteMsgException:
                    pass
                break
        # self.buffer_lock.release()

    def parse_pos_msg(self, data, socket):
        msg = UdpPosMsg(data)
        if not msg.error:
            self.cur_pos_msg = msg

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




