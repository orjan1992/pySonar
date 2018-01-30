import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from blinker import signal
import threading

from messages.sonarMsg import MtHeadData
from settings import Settings
from ogrid.oGrid import OGrid
from messages.UdpMessageClient import UdpMessageClient
from messages.moosMsgs import MoosMsgs

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

        self.settings = Settings()
        self.binary_plot_on = self.settings.grid_settings['binary_grid'] == 1

        main_layout = QtGui.QHBoxLayout() # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QGridLayout()
        bottom_right_layout = QtGui.QGridLayout()

        graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        self.plot_window = pg.PlotItem()
        graphics_view.addItem(self.plot_window)
        # IMAGE Window
        self.img_item = pg.ImageItem(autoLevels=False, levels=(-50.0, 50.0))  # image item. the actual plot
        # step = (len(self.settings.plot_colors["steps"]) - 1) / (self.settings.plot_colors["max_val"] - self.settings.plot_colors["min_val"])
        # steps = np.arange(self.settings.plot_colors["min_val"], self.settings.plot_colors["max_val"], step)
        # colors = np.zeros((len(steps), 3))
        # for i in range(0, len(steps)):
        #     colors[i, 1:3] = 0
        colormap = pg.ColorMap(self.settings.plot_colors["steps"], np.array(
            self.settings.plot_colors["colors"]))
        self.img_item.setLookupTable(colormap.getLookupTable(mode='byte'))

        self.plot_window.addItem(self.img_item)
        self.plot_window.getAxis('left').setGrid(200)
        self.img_item.getViewBox().invertY(True)

        # Button
        self.start_plotting_button = QtGui.QPushButton('Start Plotting')

        # Textbox
        self.threshold_box = QtGui.QSpinBox()
        self.threshold_box.setMinimum(0)
        self.threshold_box.setMaximum(255)
        self.threshold_box.setValue(self.settings.threshold)

        # binary plot
        self.binary_plot_button = QtGui.QPushButton('Set Prob mode')
        self.binary_plot_button.clicked.connect(self.binary_button_click)
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
        if self.settings.input_source == 0:
            self.udp_client = UdpMessageClient(self.settings.connection_settings["sonar_port"])
            client_thread = threading.Thread(target=self.udp_client.connect, daemon=True)
            client_thread.start()
        elif self.settings.input_source == 1:
            self.moos_msg_client = MoosMsgs()
            self.moos_msg_client.run()
        new_msg_signal = signal('new_msg_sonar')
        new_msg_signal.connect(self.new_msg)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1.0/24.0)




    def init_grid(self):
        self.grid = OGrid(self.settings.grid_settings["half_grid"] == 1,
                          self.settings.grid_settings["p_inital"],
                          self.settings.grid_settings["binary_threshold"])
        self.img_item.scale(self.grid.cellSize, self.grid.cellSize)

    def clear_grid(self):
        self.grid.clearGrid()
        self.plot_updated = False
        self.update_plot()

    def new_msg(self, sender, **kw):
        msg = kw["msg"]
        # msg = MtHeadData(msg)
        # msg.step *= 1.0/3200.0
        # msg.bearing *= 1.0/3200.0
        self.grid.autoUpdateZhou(msg, self.threshold_box.value())
        self.plot_updated = True
        # if self.settings.grid_settings["half_grid"] == 1:
        #     self.img_item.setPos(-msg.range_scale/10.0, -msg.range_scale/5.0)
        # else:
        #     self.img_item.setPos(-msg.range_scale / 10.0, -msg.range_scale / 10.0)
        # self.img_item.scale(16010.0/msg.range_scale, 16010.0/msg.range_scale)

    def update_plot(self):
        if self.plot_updated:
            if self.binary_plot_on:
                self.img_item.setImage(self.grid.get_binary_map().T, levels=(0, 1))
            else:
                self.img_item.setImage(self.grid.get_p().T, levels=(-5.0, 5.0))
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
