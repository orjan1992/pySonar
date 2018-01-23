import logging
from math import pi

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from settings import Settings

from ogrid.oGrid import OGrid

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


    binary_plot = True
    ogrid_conditions = [0.1, 16, 8, 0.65, 0.78]
    raw_res = 0.05
    old_pos_msg = 0
    grid = 0
    morse_running = False
    latest_sonar_msg_moose = None
    latest_pos_msg_moose = None
    old_pos_msg_moose = None
    counter = 0
    counter2 = 0

    cur_lat = 0
    cur_long = 0
    cur_head = 0


    first_run = True
    pause = True
    start_rec_on_moos_start = False
    # fname = 'logs/360 degree scan harbour piles.V4LOG' #inital file name
    fname = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
    # fname = '/home/orjangr/Repos/pySonar/logs/Sonar Log/UdpHubLog_4001_2017_11_02_09_16_58.csv'

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        self.settings = Settings()


        main_layout = QtGui.QHBoxLayout() # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QGridLayout()
        bottom_right_layout = QtGui.QGridLayout()

        graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        self.plot_window = pg.PlotItem()
        graphics_view.addItem(self.plot_window)
        # IMAGE Window
        self.img_item = pg.ImageItem(autoLevels=False, levels=(0, 1))  # image item. the actual plot
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

        # Clear grid
        self.clear_grid_button = QtGui.QPushButton('Clear Grid!')

        # Time box
        self.msg_date = QtGui.QLineEdit()
        self.msg_date.setReadOnly(True)

        self.msg_time = QtGui.QLineEdit()
        self.msg_time.setReadOnly(True)

        self.date_text = QtGui.QLineEdit()
        self.date_text.setReadOnly(True)

        #Lat long rot boxes
        self.lat_box = QtGui.QLineEdit()
        self.lat_box.setReadOnly(True)

        self.long_box = QtGui.QLineEdit()
        self.long_box.setReadOnly(True)

        self.rot_box = QtGui.QLineEdit()
        self.rot_box.setReadOnly(True)



        # Adding items
        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.start_plotting_button)
        left_layout.addWidget(self.binary_plot_button)
        left_layout.addWidget(self.clear_grid_button)

        bottom_right_layout.addWidget(self.msg_date, 0, 0, 1, 1)
        bottom_right_layout.addWidget(self.msg_time, 0, 1, 1, 1)

        bottom_right_layout.addWidget(self.lat_box, 1, 0, 1, 1)
        bottom_right_layout.addWidget(self.long_box, 1, 1, 1, 1)
        bottom_right_layout.addWidget(self.rot_box, 1, 2, 1, 1)

        right_layout.addWidget(graphics_view, 0, 0, 1, 1)
        right_layout.addLayout(bottom_right_layout, 3, 0, 1, 1)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        #register button presses
        self.start_plotting_button.clicked.connect(self.log_plotter_init)
        self.clear_grid_button.clicked.connect(self.clear_img)
        self.binary_plot_button.clicked.connect(self.binary_plot_clicked)

        #######
        self.timer = QtCore.QTimer()
        self.timer_save = QtCore.QTimer()
        self.timer_save_log = QtCore.QTimer()
        # self.colormap = plt.get_cmap('OrRd')
        self.rec_started = False

        self.timer_save.timeout.connect(self.moos_save_img)
        self.timer_save_log.timeout.connect(self.log_save_img)

    def init_grid(self):
        self.grid = Ogrid(self.settings.grid_settings["cell_size"],
                  self.settings.grid_settings["size_x"],
                  self.settings.grid_settings["size_y"],
                  self.settings.grid_settings["p_inital"],
                  self.settings.grid_settings["binary_threshold"])

    def log_plotter_init(self):
        if self.first_run:
            if self.fname.split('.')[-1] == 'csv':
                self.file = ReadCsvFile(self.fname, sonarPort =4002, posPort=13102, cont_reading=self.cont_reading_checkbox.checkState())
            else:
                self.file = ReadLogFile(self.fname)
            self.grid = OGrid(self.ogrid_conditions[0], self.ogrid_conditions[1], self.ogrid_conditions[2], self.ogrid_conditions[3])
            self.raw_grid = RawPlot(self.raw_res, self.ogrid_conditions[1], self.ogrid_conditions[2])
            self.img_item.scale(self.grid.cellSize, self.grid.cellSize)
            self.img_item.setPos(-self.grid.XLimMeters, -self.grid.YLimMeters)
            self.timer.timeout.connect(self.log_updater)
            self.first_run = False

        if self.pause:
            self.timer.start(0)
            self.pause = False
            if self.start_rec_on_moos_start:
                self.start_rec()
                self.start_rec_on_moos_start = False
            self.start_plotting_button.setText('Stop plotting ')
        else:
            self.stop_plot()

    def log_updater(self):
        msg = self.file.read_next_msg()
        updated = False
        if msg == -1:
            self.stop_plot()
        elif msg != 0:
            if msg.sensor == 2:
                while msg.type != 2:
                    msg = self.file.read_next_msg()
                self.grid.autoUpdateZhou(msg, self.threshold_box.value())
                self.raw_grid.autoUpdate(msg)
                self.raw_img_item.setImage(self.raw_grid.grid.T)
                updated = True
                self.scan_line.setData(np.arange(len(msg.data)), msg.data)
                self.new_sonar_msg = msg
            elif msg.sensor == 1:
                if not self.old_pos_msg:
                    self.old_pos_msg = msg
                delta_msg = msg - self.old_pos_msg
                if delta_msg.x != 0 or delta_msg.y != 0 or delta_msg.head != 0:
                    self.grid.translational_motion(delta_msg.y, delta_msg.x)  # ogrid reference frame
                    self.grid.rotate_grid(delta_msg.head)
                    self.lat_box.setText('Lat: {:G}'.format(msg.lat))
                    self.long_box.setText('Long: {:G}'.format(msg.long))
                    self.rot_box.setText('Heading: {:G} deg'.format(msg.head))
                    logger.debug('Delta x: {}\tDeltaY: {}\tDelta psi: {} deg'.format(delta_msg.x, delta_msg.y,
                                                                                     delta_msg.head*pi/180))

            self.msg_date.setText('Date: {}'.format(msg.date))
            self.msg_time.setText('Time: {}'.format(msg.time))
            if updated:
                if not self.binary_plot:
                    self.img_item.setImage(self.grid.getP().T)
                else:
                    self.img_item.setImage(self.grid.get_binary_map().T)

    def clear_img(self):
        self.img_item.setImage(np.zeros(np.shape(self.img_item.image)))
        if self.grid:
            self.grid.clearGrid()
            self.raw_grid.clearGrid()

    def getFile(self):
        self.stop_plot()
        self.fname = QtGui.QFileDialog.getOpenFileName(self.parent(), 'Open log file', 'logs/', "Log files (*.csv *.V4LOG)")
        self.first_run = True
        if self.grid:
            self.grid.clearGrid()
            self.clear_img()

    def stop_plot(self):
        self.timer.stop()
        self.timer_save.stop()
        self.pause = True
        self.start_plotting_button.setText('Start plotting')

    def binary_plot_clicked(self):
        self.binary_plot = not self.binary_plot
        if self.binary_plot:
            self.binary_plot_button.setText('Set Prob mode')
        else:
            self.binary_plot_button.setText('Set Binary mode')

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
