from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
from readLogFile.oGrid import OGrid
from readLogFile.sonarMsg import SonarMsg
from readLogFile.readLogFile import ReadLogFile


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        self.login_widget = LoginWidget(self)
        self.login_widget.button.clicked.connect(self.plotter)
        self.central_widget.addWidget(self.login_widget)

        log = 'logs/360 degree scan harbour piles.V4LOG'
        self.file = ReadLogFile(log)
        self.O = OGrid(0.1, 20, 15, 0.5)
        self.Threshold = 60

    def plotter(self):
        self.data =[0]
        self.curve = self.login_widget.plot.getPlotItem().image()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(0)

    def updater(self):
        msg = self.file.readNextMsg()
        while msg.type != 2:
            msg = self.file.readNextMsg()
        self.O.autoUpdateZhou(msg, self.Threshold)
        self.data = self.O.getP()
        print(self.data)
        self.curve.setData(self.data)

class LoginWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(LoginWidget, self).__init__(parent)
        layout = QtGui.QHBoxLayout()
        self.button = QtGui.QPushButton('Start Plotting')
        layout.addWidget(self.button)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.setLayout(layout)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()