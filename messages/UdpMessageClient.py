import sys
import socketserver
import io
from messages.sonarMsg import MtHeadData, UncompleteMsgException, CorruptMsgException, OtherMsgTypeException


class UdpMessageClient(socketserver.BaseRequestHandler):
    buffer = io.BytesIO()

    def __init__(self, listen_port):
        server = socketserver.UDPServer((None, listen_port), self.listener)
        server.serve_forever()

    def listener(self):
        data = self.request[0].strip()
        socket = self.request[1]
        if str.startswith('$ROV,'):
            raise NotImplementedError
        self.buffer.seek(0, io.SEEK_END)
        self.buffer.write(data)
        tmp = self.buffer.getbuffer()
        i = 0
        while tmp[i] != 0x40 and i < len(tmp):
            i = i+1
        if tmp[i] == 0x40:
            try:
                self.last_message = MtHeadData(tmp[i:len(tmp)])
                self.buffer = io.BytesIO()
            except CorruptMsgException:
                self.buffer = io.BytesIO()
            except OtherMsgTypeException:
                self.buffer = io.BytesIO()
            except UncompleteMsgException:
                pass


