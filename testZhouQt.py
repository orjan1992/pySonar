from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
from readLogFile.oGrid import OGrid
from readLogFile.sonarMsg import SonarMsg
from readLogFile.readLogFile import ReadLogFile


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.login_widget = LoginWidget(self)
        self.login_widget.button.clicked.connect(self.plotter)
        self.central_widget.addWidget(self.login_widget)
        self.setCentralWidget(self.central_widget)



    def plotter(self):
        log = 'logs/360 degree scan harbour piles.V4LOG'
        self.file = ReadLogFile(log)
        self.O = OGrid(0.1, 20, 15, 0.5)
        self.Threshold = 60
        self.img = self.login_widget.img_item
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
            self.O.autoUpdateZhou(msg, self.Threshold)
            self.img.setImage(self.O.getP().T*255)


class LoginWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(LoginWidget, self).__init__(parent)
        layout = QtGui.QHBoxLayout() # Main layout
        self.button = QtGui.QPushButton('Start Plotting')
        layout.addWidget(self.button) # adding button to main widget
        self.graphics_view = pg.GraphicsLayoutWidget() # layout for holding graphics object
        view_box = pg.ViewBox(invertY=True) # making viewbox for the image, inverting y to make it right
        self.img_item = pg.ImageItem() # image item. the actual plot
        view_box.addItem(self.img_item)
        self.graphics_view.addItem(view_box)
        layout.addWidget(self.graphics_view)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
