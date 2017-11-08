from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
from readLogFile.oGrid import OGrid
from readLogFile.sonarMsg import SonarMsg
from readLogFile.readLogFile import ReadLogFile
import numpy as np


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

        graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        view_box = pg.ViewBox(invertY=True) # making viewbox for the image, inverting y to make it right

        # IMAGE Window
        self.img_item = pg.ImageItem() # image item. the actual plot
        colormap = pg.ColorMap([0, 0.33, 0.67, 1], np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))
        self.img_item.setLookupTable(colormap.getLookupTable(mode='byte'))

        # Button
        self.button = QtGui.QPushButton('Start Plotting')

        # Textbox
        self.threshold_box = QtGui.QSpinBox()
        self.threshold_box.setMinimum(0)
        self.threshold_box.setMaximum(255)
        self.threshold_box.setValue(60)

        # Select file
        self.select_file_button = QtGui.QPushButton('Select File')


        # Adding items
        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.button)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(graphics_view)

        view_box.addItem(self.img_item)
        graphics_view.addItem(view_box)
        self.setLayout(main_layout)
        self.button.clicked.connect(self.plotter)

    def plotter(self):
        log = 'logs/360 degree scan harbour piles.V4LOG'
        self.file = ReadLogFile(log)
        self.O = OGrid(0.1, 20, 15, 0.5)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(0)

    def updater(self):
        msg = self.file.readNextMsg()
        if msg == -1:
            self.timer.stop()
        if msg != 0:
            while msg.type != 2:
                msg = self.file.readNextMsg()
            self.O.autoUpdateZhou(msg, self.threshold_box.value())
            self.img_item.setImage(self.O.getP().T)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
