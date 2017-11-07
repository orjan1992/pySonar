import numpy as np
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)


from PyQt4 import QtGui  # (the example applies equally well to PySide)
import pyqtgraph as pg

## Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])

## Define a top-level widget to hold everything
w = QtGui.QWidget()

## Create some widgets to be placed inside
plot = pg.PlotWidget()
plot.plot(x, y, pen=None, symbol='o')  ## setting pen=None disables line drawing
## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

## Add widgets to the layout in their proper positions

layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows

## Display the widget as a new window
w.show()

## Start the Qt event loop
app.exec_()