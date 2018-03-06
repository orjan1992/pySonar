import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt

app = QtGui.QApplication([])
w = pg.PlotWindow()


im = np.load('test.npz')['olog'].astype(np.uint8)
hist, edges = np.histogram(im, 256)
# blur = cv2.GaussianBlur(im, (7, 7), 0)

bar = pg.BarGraphItem(x=np.arange(256), y1=hist, width=0.3, brush='r')

w.addItem(bar)

w.show()

app.exec()