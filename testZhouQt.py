import logging
from math import pi

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtCore, QtGui

from ogrid.oGrid import OGrid
from readLogFile.readCsvFile import ReadCsvFile
from readLogFile.readLogFile import ReadLogFile

LOG_FILENAME = 'logging_example.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,)


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.login_widget = MainWidget(self)
        self.central_widget.addWidget(self.login_widget)
        self.setCentralWidget(self.central_widget)


class MainWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        main_layout = QtGui.QHBoxLayout() # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QVBoxLayout()
        bottom_right_layout = QtGui.QHBoxLayout()

        graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        view_box = pg.ViewBox(invertY=True) # making viewbox for the image, inverting y to make it right

        # IMAGE Window
        self.img_item = pg.ImageItem() # image item. the actual plot
        colormap = pg.ColorMap([0, 0.33, 0.67, 1], np.array([[0.2, 0.2, 0.2, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))
        self.img_item.setLookupTable(colormap.getLookupTable(mode='byte'))

        # Button
        self.start_plotting_button = QtGui.QPushButton('Start Plotting')

        # Textbox
        self.threshold_box = QtGui.QSpinBox()
        self.threshold_box.setMinimum(0)
        self.threshold_box.setMaximum(255)
        self.threshold_box.setValue(60)

        # Select file
        self.select_file_button = QtGui.QPushButton('Select File')

        # Time box
        self.msg_date = QtGui.QLineEdit()
        self.msg_date.setReadOnly(True)
        self.msg_time = QtGui.QLineEdit()
        self.msg_time.setReadOnly(True)

        # Adding items
        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.start_plotting_button)
        left_layout.addWidget(self.select_file_button)

        view_box.addItem(self.img_item)
        graphics_view.addItem(view_box)

        bottom_right_layout.addWidget(self.msg_date)
        bottom_right_layout.addWidget(self.msg_time)

        right_layout.addWidget(graphics_view)
        right_layout.addLayout(bottom_right_layout)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        #register button presses
        self.start_plotting_button.clicked.connect(self.plotter_init)
        self.select_file_button.clicked.connect(self.getFile)

        #######
        self.first_run = True
        self.pause = True
        self.fname = 'logs/360 degree scan harbour piles.V4LOG' #inital file name
        # self.fname = 'logs/UdpHubLog_4001_2017_11_02_09_01_58.csv'
        self.timer = QtCore.QTimer()
        self.ogrid_conditions = [0.1, 20, 15, 0.5]
        self.old_pos_msg = 0
        self.grid = 0

    def plotter_init(self):
        if self.first_run:
            if self.fname.split('.')[-1] == 'csv':
                self.file = ReadCsvFile(self.fname)
            else:
                self.file = ReadLogFile(self.fname)
            self.grid = OGrid(self.ogrid_conditions[0], self.ogrid_conditions[1], self.ogrid_conditions[2], self.ogrid_conditions[3])
            self.timer.timeout.connect(self.updater)
            self.first_run = False

        if self.pause:
            self.timer.start(0)
            self.pause = False
            self.start_plotting_button.setText('Stop plotting ')
        else:
            self.stop_plot()

    def updater(self):
        msg = self.file.readNextMsg()
        if msg == -1:
            self.stop_plot()
        elif msg != 0:
            if msg.sensor == 2:
                while msg.type != 2:
                    msg = self.file.readNextMsg()
                self.grid.autoUpdateZhou(msg, self.threshold_box.value())
                self.img_item.setImage(self.grid.getP().T)
            elif msg.sensor == 1:
                if not self.old_pos_msg:
                    self.old_pos_msg = msg
                self.grid.rot_motion(msg.head - self.old_pos_msg.head)
                self.grid.translational_motion(msg.x - self.old_pos_msg.x, msg.y - self.old_pos_msg.y)
                print('Delta x: {}\nDeltaY: {}\nDelta psi: {}'.format(msg.x - self.old_pos_msg.x, msg.y - self.old_pos_msg.y, (msg.head - self.old_pos_msg.head)*180/pi))

            self.msg_date.setText(msg.date)
            self.msg_time.setText(msg.time)

    def clear_img(self):
        self.img_item.setImage(np.zeros(np.shape(self.img_item.image)))

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


if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
