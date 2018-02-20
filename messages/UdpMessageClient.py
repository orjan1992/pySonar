import sys
import socketserver
import io
from messages.sonarMsg import MtHeadData, UncompleteMsgException, CorruptMsgException, OtherMsgTypeException
from blinker import signal


class UdpMessageClient(object):
    buffer = io.BytesIO()

    def __init__(self, listen_port, sonar_msg_callback):
        self.new_msg_signal = signal('new_msg_sonar')
        self.listen_port = listen_port
        signal('new_udp_msg').connect(self.parse_msg)
        self.sonar_msg_callback = sonar_msg_callback

    def connect(self):
        server = socketserver.UDPServer(('0.0.0.0', self.listen_port), UdpMessageHandler)
        server.serve_forever()

    def parse_msg(self, sender, **args):
        data = args["data"]
        try:
            tmp = data[0:5].decode('ascii')
            if tmp == '$ROV,':
                raise NotImplementedError
        except Exception:
            print("Data to short")
        self.buffer.seek(0, io.SEEK_END)
        self.buffer.write(data)
        tmp = self.buffer.getbuffer()
        for i in range(0, len(tmp)):
            if tmp[i] == 0x40:
                try:
                    last_message = MtHeadData(tmp[i:len(tmp)])
                    # self.new_msg_signal.send(self, msg=last_message)
                    self.sonar_msg_callback(last_message)
                    self.buffer = io.BytesIO()
                except CorruptMsgException:
                    self.buffer = io.BytesIO()
                except OtherMsgTypeException:
                    self.buffer = io.BytesIO()
                    print("Other msg type")
                except UncompleteMsgException:
                    pass
                break


class UdpMessageHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = self.request[0].strip()
        socket = self.request[1]
        signal('new_udp_msg').send(self, data=data)


