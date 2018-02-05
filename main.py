import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from blinker import signal
import threading
import sys

from messages.sonarMsg import MtHeadData
from settings import *
from ogrid.oGrid import OGrid
from messages.UdpMessageClient import UdpMessageClient
from messages.moosMsgs import MoosMsgs
from messages.moosPosMsg import *

LOG_FILENAME = 'main.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)
logger = logging.getLogger('main')
WINDOWS = True


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.login_widget = MainWidget(self)
        self.central_widget.addWidget(self.login_widget)
        self.setCentralWidget(self.central_widget)


class MainWidget(QtGui.QWidget):
    plot_updated = False
    grid = None

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        if Settings.input_source == 0:
            # TODO: Do something
            self.last_pos_msg = None
        elif Settings.input_source == 1:
            self.last_pos_msg = MoosPosMsg()
            self.last_pos_diff = MoosPosMsgDiff(0, 0, 0)

        main_layout = QtGui.QHBoxLayout() # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QGridLayout()
        bottom_right_layout = QtGui.QGridLayout()

        graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        self.plot_window = pg.PlotItem()
        graphics_view.addItem(self.plot_window)
        # IMAGE Window
        self.img_item = pg.ImageItem(autoLevels=False)

        colormap = pg.ColorMap(PlotSettings.steps, np.array(
            PlotSettings.colors))
        self.img_item.setLookupTable(colormap.getLookupTable(mode='byte'))

        self.plot_window.addItem(self.img_item)
        self.plot_window.getAxis('left').setGrid(200)
        self.img_item.getViewBox().invertY(True)
        self.img_item.setOpts(axisOrder='row-major')

        # Button
        self.start_plotting_button = QtGui.QPushButton('Start Plotting')

        # Textbox
        self.threshold_box = QtGui.QSpinBox()
        self.threshold_box.setMinimum(0)
        self.threshold_box.setMaximum(255)
        self.threshold_box.setValue(PlotSettings.threshold)

        # binary plot
        self.binary_plot_button = QtGui.QPushButton('Set Prob mode')
        self.binary_plot_button.clicked.connect(self.binary_button_click)
        if Settings.raw_plot:
            self.binary_plot_on = False
        else:
            self.binary_plot_on = GridSettings.binary_grid
        if not self.binary_plot_on:
            self.binary_plot_button.text = "Set Binary mode"

        # Clear grid
        self.clear_grid_button = QtGui.QPushButton('Clear Grid!')
        self.clear_grid_button.clicked.connect(self.clear_grid)




        # Adding items
        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.start_plotting_button)
        left_layout.addWidget(self.binary_plot_button)
        left_layout.addWidget(self.clear_grid_button)

        right_layout.addWidget(graphics_view, 0, 0, 1, 1)
        right_layout.addLayout(bottom_right_layout, 3, 0, 1, 1)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        self.init_grid()
        if Settings.input_source == 0:
            self.udp_client = UdpMessageClient(ConnectionSettings.sonar_port)
            client_thread = threading.Thread(target=self.udp_client.connect, daemon=True)
            client_thread.start()
        elif Settings.input_source == 1:
            self.moos_msg_client = MoosMsgs()
            self.moos_msg_client.run()
        else:
            raise Exception('Uknown input source')
        new_msg_signal = signal('new_msg_sonar')
        new_msg_signal.connect(self.new_sonar_msg)
        new_pos_msg_signal = signal('new_msg_pos')
        new_pos_msg_signal.connect(self.new_pos_msg)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1.0/24.0)




    def init_grid(self):
        self.grid = OGrid(GridSettings.half_grid,
                          GridSettings.p_inital,
                          GridSettings.binary_threshold)
        self.img_item.scale(self.grid.cellSize, self.grid.cellSize)

    def clear_grid(self):
        self.grid.clearGrid()
        self.plot_updated = False
        self.update_plot()

    def new_sonar_msg(self, sender, **kw):
        msg = kw["msg"]
        self.grid.update_distance(msg.range_scale)
        if Settings.raw_plot:
            self.grid.update_raw(msg)
        else:
            self.grid.auto_update_zhou(msg, self.threshold_box.value())
        self.plot_updated = True
        # if self.settings.grid_settings["half_grid"] == 1:
        #     self.img_item.setPos(-msg.range_scale/10.0, -msg.range_scale/5.0)
        # else:
        #     self.img_item.setPos(-msg.range_scale / 10.0, -msg.range_scale / 10.0)
        # self.img_item.scale(16010.0/msg.range_scale, 16010.0/msg.range_scale)

    def new_pos_msg(self, sender, **kw):
        msg = kw["msg"]
        diff = (msg - self.last_pos_msg) + self.last_pos_diff
        if diff.big_difference:
            self.grid.translational_motion(diff.dx, diff.dy)
            self.grid.rotate_grid(diff.dpsi)
            self.last_pos_diff = MoosPosMsgDiff(0, 0, 0)
            self.plot_updated = True
        else:
            self.last_pos_diff = diff
        self.last_pos_msg = msg



    def update_plot(self):
        if self.plot_updated:
            if Settings.raw_plot:
                self.img_item.setImage(self.grid.get_raw(), levels=(0.0, 255.0))
            else:
                if self.binary_plot_on:
                    self.img_item.setImage(self.grid.get_binary_map(), levels=(0, 1))
                else:
                    self.img_item.setImage(self.grid.get_p(), levels=(-5.0, 5.0))
            self.img_item.setPos(-self.grid.last_distance, -self.grid.last_distance/2 if GridSettings.half_grid else -self.grid.last_distance)
            # self.img_item.scale(self.grid.last_distance/self.grid.RES, self.grid.last_distance/self.grid.RES)
            self.plot_updated = False

    def binary_button_click(self):
        if self.binary_plot_on:
            self.binary_plot_button.text = "Set Prob mode"
        else:
            self.binary_plot_button.text = "Set Binary mode"
        self.binary_plot_on = not self.binary_plot_on




if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
