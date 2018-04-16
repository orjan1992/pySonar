import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from messages.udpMsg import *
from messages.udpClient_py import *
from scipy import signal

p1 = pg.plot()
curve = p1.getPlotItem()

def plot(msg):
    # global curve
    # curve.setData(msg.data)
    global p1
    p1.plotItem.plot(msg.data, clear=True)


if __name__ == '__main__':
    udp_client = UdpClient(4001, 4005, None, None)
    udp_client.set_sonar_callback(plot)
    udp_client.start()
    app = QtGui.QApplication([])

    p1.setYRange(0, 255)
    # p1.autoPixelRange = False
    p1.show()

    app.exec_()