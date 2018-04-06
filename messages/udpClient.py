import re
import io
from messages.udpMsg import *
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtNetwork import *
from settings import ConnectionSettings
import logging
logger = logging.getLogger('UdpClient')


class UdpClient(QObject):
    buffer = io.BytesIO()

    def __init__(self, ip, port):
        super().__init__(None)
        self.port = port
        self.ip = ip
        self.server = QUdpSocket()
        if ip is None:
            self.server.bind(port)
        else:
            self.server.bind(ip, port)


class UdpSonarClient(UdpClient):
    signal_new_sonar_msg = pyqtSignal(object, name='new_sonar_msg')

    def __init__(self, ip, port):
        super().__init__(ip, port)
        self.server.readyRead.connect(self.parse_msg)

    @pyqtSlot()
    def parse_msg(self):
        datagram = self.server.receiveDatagram()
        self.buffer.write(datagram.data())
        tmp = self.buffer.getbuffer()
        for i in range(0, len(tmp)):
            if tmp[i] == 0x40:
                try:
                    msg = MtHeadData(tmp[i:len(tmp)])
                    # self.new_msg_signal.send(self, msg=last_message)
                    self.signal_new_sonar_msg.emit(msg)
                    self.buffer = io.BytesIO()
                except CorruptMsgException:
                    self.buffer = io.BytesIO()
                except OtherMsgTypeException:
                    self.buffer = io.BytesIO()
                    print("Other msg type")
                except UncompleteMsgException:
                    pass
                break


class UdpNmeaClient(UdpClient):
    signal_new_pos_msg = pyqtSignal(object, name='new_sonar_msg')
    cur_pos_msg = None

    def __init__(self, ip, port):
        super().__init__(ip, port)
        self.server.readyRead.connect(self.parse_msg)

    @pyqtSlot()
    def parse_msg(self):
        datagram = self.server.receiveDatagram()
        # string = datagram.data().decode('ascii')
        data = bytearray(datagram.data())
        self.cur_pos_msg = UdpPosMsg(data)

    def sendMsg(value):
        datagram = QNetworkDatagram()
        raise NotImplemented






