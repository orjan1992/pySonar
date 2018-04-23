import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from messages.udpMsg import *
from messages.moosMsgs import *
from messages.udpClient_py import *
from scipy import signal



class haha(QObject):
    scan_list = []
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
        self.client = MoosMsgs()
        self.client.signal_new_sonar_msg.connect(self.plot)
        self.client.run()


    @QtCore.pyqtSlot(object, name='new_sonar_msg')
    def plot(self, msg):
        # global curve
        # curve.setData(msg.data)
        self.p1.plotItem.plot(msg.data, clear=True)
        self.scan_list.append(msg.data)


if __name__ == '__main__':
    global scan_list
    # udp_client = UdpClient(4001, 4005, None, None)
    # udp_client.set_sonar_callback(plot)
    # udp_client.start()
    p1 = pg.plot()
    b = haha(p1)

    app = QtGui.QApplication([])

    p1.setYRange(0, 255)
    # p1.autoPixelRange = False
    p1.show()

    app.exec_()
    np.savez('scanline.npz', scanline=np.array(b.scan_list))