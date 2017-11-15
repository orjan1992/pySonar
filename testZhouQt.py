import logging
from math import pi

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtCore, QtGui

from ogrid.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.readLogFile import ReadLogFile
from messages.moosMsgs import MoosMsgs

LOG_FILENAME = 'ZhouLog.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)
logger = logging.getLogger('ZhouQt')
logger.disabled = True
# logging.getLogger('readLogFile.ReadCsvFile').disabled = True
logging.getLogger('messages.MoosMsgs').disabled = True
logging.getLogger('messages.MoosMsgs.bins').disabled = True
logging.getLogger('messages.MoosMsgs.pose').disabled = True
# logging.getLogger('readLogFile.ReadCsvFile').disabled = True


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.login_widget = MainWidget(self)
        self.central_widget.addWidget(self.login_widget)
        self.setCentralWidget(self.central_widget)


class MainWidget(QtGui.QWidget):
    binary_plot = True
    ogrid_conditions = [0.3, 60, 30, 0.5]
    old_pos_msg = 0
    grid = 0
    morse_running = False
    latest_sonar_msg_moose = None
    latest_pos_msg_moose = None
    old_pos_msg_moose = None
    counter = 0


    first_run = True
    pause = True
    # fname = 'logs/360 degree scan harbour piles.V4LOG' #inital file name
    # fname = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
    fname = '/home/orjangr/Repos/pySonar/logs/Sonar Log/UdpHubLog_4001_2017_11_02_09_16_58.csv'

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        main_layout = QtGui.QHBoxLayout() # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QVBoxLayout()
        bottom_right_layout = QtGui.QGridLayout()

        graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        # view_box = pg.ViewBox(invertY=True) # making viewbox for the image, inverting y to make it right
        self.plot_window = pg.PlotItem()
        graphics_view.addItem(self.plot_window)
        # IMAGE Window
        self.img_item = pg.ImageItem(autoLevels=False, levels=(0, 1))  # image item. the actual plot
        colormap = pg.ColorMap([0, 0.33, 0.67, 1], np.array(
            [[0.2, 0.2, 0.2, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))
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
        self.threshold_box.setValue(60)

        # Select file
        self.select_file_button = QtGui.QPushButton('Select File')
        # From Morse
        self.from_morse_button = QtGui.QPushButton('From Morse')
        # Check box for continious reading
        self.cont_reading_checkbox = QtGui.QCheckBox()
        self.cont_reading_checkbox.setText('Read multiple files')
        self.cont_reading_checkbox.setChecked(True)

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
        left_layout.addWidget(self.select_file_button)
        left_layout.addWidget(self.cont_reading_checkbox)
        left_layout.addWidget(self.from_morse_button)
        left_layout.addWidget(self.binary_plot_button)
        left_layout.addWidget(self.clear_grid_button)

        # view_box.addItem(self.img_item)
        # graphics_view.addItem(view_box)

        bottom_right_layout.addWidget(self.msg_date, 0, 0, 1, 1)
        bottom_right_layout.addWidget(self.msg_time, 0, 1, 1, 1)

        bottom_right_layout.addWidget(self.lat_box, 1, 0, 1, 1)
        bottom_right_layout.addWidget(self.long_box, 1, 1, 1, 1)
        bottom_right_layout.addWidget(self.rot_box, 1, 2, 1, 1)

        right_layout.addWidget(graphics_view)
        right_layout.addLayout(bottom_right_layout)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        #register button presses
        self.start_plotting_button.clicked.connect(self.log_plotter_init)
        self.select_file_button.clicked.connect(self.getFile)
        self.from_morse_button.clicked.connect(self.run_morse)
        self.clear_grid_button.clicked.connect(self.clear_img)
        self.binary_plot_button.clicked.connect(self.binary_plot_clicked)

        #######
        self.timer = QtCore.QTimer()


    def run_morse(self):
        if self.first_run:
            self.grid = OGrid(self.ogrid_conditions[0], self.ogrid_conditions[1], self.ogrid_conditions[2],
                              self.ogrid_conditions[3])
            self.img_item.scale(self.grid.cellSize, self.grid.cellSize)
            self.img_item.setPos(-self.grid.XLimMeters, -self.grid.YLimMeters)
            self.first_run = False
        else:
            self.grid.clearGrid()
        if self.morse_running:
            self.moos_client.close()
            self.morse_running = False
            self.stop_plot()
            self.from_morse_button.setText('From Morse')
        else:
            self.moos_client = MoosMsgs()
            self.moos_client.set_on_sonar_msg_callback(self.moos_sonar_message_recieved)
            self.moos_client.set_on_pos_msg_callback(self.moos_pos_message_recieved)
            self.moos_client.run()
            self.morse_running = True
            self.timer.timeout.connect(self.moos_updater)
            self.timer.start(0)
            self.pause = False
            self.from_morse_button.setText('Stop from morse ')

    def moos_sonar_message_recieved(self, msg):
        msg.rangeScale = 30
        msg.step = 1.8*pi/180
        logger.info('max = {}'.format(max(msg.data)))
        msg.bearing = msg.bearing-pi/2
        self.latest_sonar_msg_moose = msg
        return True

    def moos_pos_message_recieved(self, msg):
        self.latest_pos_msg_moose = msg
        return True

    def moos_updater(self):
        updated = False
        # self.moos_client.send_speed(0.1, 0.001)
        if self.latest_sonar_msg_moose:
            self.grid.autoUpdateZhou(self.latest_sonar_msg_moose, self.threshold_box.value())
            updated = True
            self.latest_sonar_msg_moose = None
        if self.latest_pos_msg_moose:
            if not self.old_pos_msg_moose:
                self.old_pos_msg_moose = self.latest_pos_msg_moose
            delta_msg = self.latest_pos_msg_moose - self.old_pos_msg_moose
            if delta_msg.x != 0 or delta_msg.y != 0 or delta_msg.head != 0:
                self.grid.translational_motion(delta_msg.y, delta_msg.x) # ogrid reference frame
                self.grid.rotate_grid(delta_msg.head)
                self.lat_box.setText('Lat: {:G}'.format(self.latest_pos_msg_moose.lat))
                self.long_box.setText('Long: {:G}'.format(self.latest_pos_msg_moose.long))
                self.rot_box.setText('Heading: {:G} deg'.format(self.latest_pos_msg_moose.head * 180 / pi))
                # print('Transformed grid: delta_x: {}\tdelta_y: {}\tdelta_psi: {} deg'.format(delta_msg.x, delta_msg.y,
                #                                                                              delta_msg.head * 180 / pi))
                updated = True
            self.old_pos_msg_moose = self.latest_pos_msg_moose
            self.latest_pos_msg_moose = None

        if updated:
            if not self.binary_plot:
                self.counter += 1
                if self.counter == 5:
                    self.img_item.setImage(self.grid.getP().T)
                    self.counter = 0
            else:
                self.img_item.setImage(self.grid.get_binary_map().T)

    def log_plotter_init(self):
        if self.first_run:
            if self.fname.split('.')[-1] == 'csv':
                self.file = ReadCsvFile(self.fname, sonarPort =4002, posPort=13102, cont_reading=self.cont_reading_checkbox.checkState())
            else:
                self.file = ReadLogFile(self.fname)
            self.grid = OGrid(self.ogrid_conditions[0], self.ogrid_conditions[1], self.ogrid_conditions[2], self.ogrid_conditions[3])
            self.img_item.scale(self.grid.cellSize, self.grid.cellSize)
            self.img_item.setPos(-self.grid.XLimMeters, -self.grid.YLimMeters)
            self.timer.timeout.connect(self.log_updater)
            self.first_run = False

        if self.pause:
            self.timer.start(0)
            self.pause = False
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
                updated = True
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

    def getFile(self):
        self.stop_plot()
        self.fname = QtGui.QFileDialog.getOpenFileName(self.parent(), 'Open log file', 'logs/', "Log files (*.csv *.V4LOG)")
        self.first_run = True
        if self.grid:
            self.grid.clearGrid()
            self.clear_img()

    def stop_plot(self):
        self.timer.stop()
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
