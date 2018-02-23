import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from blinker import signal
import threading
from copy import deepcopy

from settings import *
from ogrid.oGrid import OGrid
from messages.UdpMessageClient import UdpMessageClient
from messages.moosMsgs import MoosMsgs
from messages.moosPosMsg import *
from collisionAvoidance import CollisionAvoidance

LOG_FILENAME = 'main.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)
logger = logging.getLogger('main')
logging.getLogger('messages.MoosMsgs.pose').disabled = True
logging.getLogger('messages.MoosMsgs.bins').disabled = True
logging.getLogger('messages.MoosMsgs.pose').disabled = True
logging.getLogger('moosPosMsg').disabled = True

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.login_widget = MainWidget(self)
        self.central_widget.addWidget(self.login_widget)
        self.setCentralWidget(self.central_widget)


# noinspection PyUnresolvedReferences
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

        main_layout = QtGui.QHBoxLayout()  # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QGridLayout()
        bottom_right_layout = QtGui.QGridLayout()

        graphics_view = pg.GraphicsLayoutWidget()  # layout for holding graphics object
        self.plot_window = pg.PlotItem()
        graphics_view.addItem(self.plot_window)
        # IMAGE Window
        self.img_item = pg.ImageItem(autoLevels=False)

        if Settings.plot_type != 2:
            colormap = pg.ColorMap(PlotSettings.steps, np.array(
                PlotSettings.colors))
            self.img_item.setLookupTable(colormap.getLookupTable(mode='byte'))

        self.plot_window.addItem(self.img_item)
        self.plot_window.getAxis('left').setGrid(200)
        self.img_item.getViewBox().invertY(True)
        self.img_item.getViewBox().setAspectLocked(True)
        self.img_item.setOpts(axisOrder='row-major')

        # Textbox
        self.threshold_box = QtGui.QSpinBox()
        self.threshold_box.setMinimum(0)
        self.threshold_box.setMaximum(255)
        self.threshold_box.setValue(PlotSettings.threshold)

        # binary plot
        self.binary_plot_button = QtGui.QPushButton('Set Prob mode')
        self.binary_plot_button.clicked.connect(self.binary_button_click)
        # if Settings.raw_plot:
        #     self.binary_plot_on = False
        # else:
        #     self.binary_plot_on = GridSettings.binary_grid
        self.binary_plot_on = GridSettings.binary_grid
        if not self.binary_plot_on:
            self.binary_plot_button.text = "Set Binary mode"

        # Clear grid button
        self.clear_grid_button = QtGui.QPushButton('Clear Grid!')
        self.clear_grid_button.clicked.connect(self.clear_grid)

        # Adding map
        if Settings.collision_avoidance and Settings.show_map:
            if Settings.input_source == 1:
                self.map_scene = QtWidgets.QGraphicsScene()
                if MapSettings.display_grid:
                    xmin = -310
                    xmax = 420
                    ymin = -200
                    ymax = 400
                    for i in range(xmin, xmax, MapSettings.grid_dist):
                        self.map_scene.addLine(i, ymin, i, ymax, MapSettings.grid_pen)
                    for i in range(ymin, ymax, MapSettings.grid_dist):
                        self.map_scene.addLine(xmin, i, xmax, i, MapSettings.grid_pen)
                    self.map_scene.addLine(-MapSettings.grid_dist, 0, MapSettings.grid_dist,
                                           0, MapSettings.grid_center_pen)
                    self.map_scene.addLine(0, -MapSettings.grid_dist, 0, MapSettings.grid_dist,
                                           MapSettings.grid_center_pen)

                self.map_scene.addEllipse(108.0412 - 5, -133.7201 - 5, 10, 10,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)
                self.map_scene.addEllipse(-53.5571 - 24.88/2, -60.9444 - 5, 24.88, 10,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)
                self.map_scene.addEllipse(-214.7458 - 5, 37.2886 - 40, 10, 80,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)

                self.map_scene.addRect(101.6381 - 7.5, 31.0354 - 13.3/2, 15, 13.3,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)

                self.map_scene.addRect(-2.0295 - 5, 120.6624 - 5, 10, 10,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)
                self.map_scene.addRect(311.4198 - 5, 120.6624 - 50, 10, 100,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)
                self.map_scene.addRect(59.9079 - 100, 406.9405 - 5, 200, 10,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)
                self.map_scene.addRect(-2.0295 - 50, -211.9193 - 5, 100, 10,
                                          MapSettings.obstacle_pen, MapSettings.obstacle_brush)

                self.map_pos_ellipse = QtWidgets.QGraphicsEllipseItem(0, 0, MapSettings.vehicle_size,
                                                                      MapSettings.vehicle_size)
                self.map_pos_ellipse.setPen(MapSettings.vehicle_pen)
                self.map_scene.addItem(self.map_pos_ellipse)

                self.map_waypoint_objects = []
                self.map_obstacle_list = []

                self.map_view = QtWidgets.QGraphicsView()
                self.map_view.setScene(self.map_scene)

        # Adding items
        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.binary_plot_button)
        left_layout.addWidget(self.clear_grid_button)

        right_layout.addWidget(graphics_view, 0, 0, 1, 1)
        right_layout.addLayout(bottom_right_layout, 3, 0, 1, 1)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        if Settings.collision_avoidance and Settings.show_map:
            main_layout.addWidget(self.map_view)
        self.setLayout(main_layout)

        if Settings.hist_window:
            self.hist_window = pg.PlotWindow()
            self.histogram = pg.BarGraphItem(x=np.arange(10), y1=np.random.rand(10), width=0.3, brush='r')
            self.hist_window.addItem(self.histogram)
            self.hist_window.show()

        self.init_grid()
        if Settings.input_source == 0:
            self.udp_client = UdpMessageClient(ConnectionSettings.sonar_port, self.new_sonar_msg)
            client_thread = threading.Thread(target=self.udp_client.connect, daemon=True)
            client_thread.start()
        elif Settings.input_source == 1:
            self.moos_msg_client = MoosMsgs(self.new_sonar_msg)
            self.moos_msg_client.run()
        else:
            raise Exception('Uknown input source')

        self.last_pos_msg = None
        self.processing_pos_msg = False
        self.pos_lock = threading.Lock()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.pos_update_timer = QtCore.QTimer()
        self.pos_update_timer.timeout.connect(self.new_pos_msg)

        if Settings.plot_type == 2:
            self.timer.start(500.0)
        else:
            self.timer.start(1000.0/24.0)

        if Settings.collision_avoidance:
            if Settings.input_source == 1:
                self.collision_avoidance = CollisionAvoidance(self.moos_msg_client)
            else:
                raise NotImplemented
            if Settings.input_source == 1:
                self.moos_msg_client.set_waypoints_callback(self.collision_avoidance.callback)
            else:
                raise NotImplemented()
            self.collision_avoidance_timer = QtCore.QTimer()
            self.collision_avoidance_timer.timeout.connect(self.collision_avoidance_loop)
            self.collision_avoidance_timer.start(Settings.collision_avoidance_interval)

        self.pos_update_timer.start(Settings.pos_update)

    def init_grid(self):
        self.grid = OGrid(GridSettings.half_grid,
                          GridSettings.p_inital,
                          GridSettings.binary_threshold)

    def clear_grid(self):
        self.grid.clear_grid()
        self.plot_updated = False
        self.update_plot()

    def new_sonar_msg(self, msg):
        self.grid.update_distance(msg.range_scale)
        if Settings.update_type == 0:
            self.grid.update_raw(msg)
        elif Settings.update_type == 1:
            self.grid.auto_update_zhou(msg, self.threshold_box.value())
        else:
            raise Exception('Invalid update type')
        self.plot_updated = True

    def new_pos_msg(self):
        if self.pos_lock.acquire(blocking=False):
            if Settings.input_source == 0:
                # TODO: fix this
                msg = MoosPosMsg()
            else:
                msg = self.moos_msg_client.cur_pos_msg
            if self.last_pos_msg is None:
                self.last_pos_msg = deepcopy(msg)

            if Settings.collision_avoidance:
                self.collision_avoidance.update_pos(msg.lat, msg.long, msg.psi)
                if Settings.show_map:
                    # Draw pos
                    self.map_scene.removeItem(self.map_pos_ellipse)
                    # self.map_pos_ellipse = \
                    #     QtWidgets.QGraphicsEllipseItem(msg.long*10 - MapSettings.vehicle_size / 2,
                    #                                    -msg.lat*10 - MapSettings.vehicle_size / 2,
                    #                                    MapSettings.vehicle_size, MapSettings.vehicle_size)
                    self.map_pos_ellipse = \
                        QtWidgets.QGraphicsRectItem(msg.long*10 -
                                                    (MapSettings.vehicle_size / 2)*MapSettings.vehicle_form_factor,
                                                       -msg.lat*10 - MapSettings.vehicle_size / 2,
                                                       MapSettings.vehicle_size*MapSettings.vehicle_form_factor,
                                                    MapSettings.vehicle_size)
                    self.map_pos_ellipse.setTransformOriginPoint(msg.long*10, -msg.lat*10)
                    self.map_pos_ellipse.setRotation(msg.psi*180.0/pi)
                    self.map_pos_ellipse.setPen(MapSettings.vehicle_pen)
                    self.map_pos_ellipse.setBrush(MapSettings.vehicle_brush)
                    self.map_scene.addItem(self.map_pos_ellipse)

                    try:
                        # remove old waypoints
                        for obj in self.map_waypoint_objects:
                            self.map_scene.removeItem(obj)
                        self.map_waypoint_objects.clear()
                    except Exception as e:
                        print('Remove waypoints: {}'.format(e))

                    # draw new
                    waypoints = self.collision_avoidance.waypoint_list
                    if len(waypoints) > 0:
                        p = QtWidgets.QGraphicsEllipseItem(waypoints[0][1] * 10 - MapSettings.waypoint_size / 2,
                                                           -waypoints[0][0] * 10 - MapSettings.waypoint_size / 2,
                                                           MapSettings.waypoint_size,
                                                           MapSettings.waypoint_size)
                        if self.collision_avoidance.waypoint_counter == 1:
                            p.setPen(MapSettings.waypoint_active_pen)
                        else:
                            p.setPen(MapSettings.waypoint_inactive_pen)
                        self.map_scene.addItem(p)
                        self.map_waypoint_objects.append(p)
                        try:
                            for i in range(1, len(waypoints)):
                                p = QtWidgets.QGraphicsEllipseItem(waypoints[i][1]*10 - MapSettings.waypoint_size / 2,
                                                                   -waypoints[i][0]*10 - MapSettings.waypoint_size / 2,
                                                                   MapSettings.waypoint_size,
                                                                   MapSettings.waypoint_size)
                                if self.collision_avoidance.waypoint_counter == i or \
                                        self.collision_avoidance.waypoint_counter == i + 1:
                                    p.setPen(MapSettings.waypoint_active_pen)
                                else:
                                    p.setPen(MapSettings.waypoint_inactive_pen)
                                self.map_scene.addItem(p)
                                self.map_waypoint_objects.append(p)

                                l = QtWidgets.QGraphicsLineItem(waypoints[i][1]*10,
                                                                -waypoints[i][0]*10,
                                                                waypoints[i - 1][1]*10,
                                                                -waypoints[i - 1][0]*10)
                                if self.collision_avoidance.waypoint_counter == i:
                                    l.setPen(MapSettings.waypoint_active_pen)
                                else:
                                    l.setPen(MapSettings.waypoint_inactive_pen)
                                self.map_scene.addItem(l)
                                self.map_waypoint_objects.append(l)
                            # TODO: Paint obstacles
                        except Exception as e:
                            print('Add waypoints: {}'.format(e))

            diff = (msg - self.last_pos_msg)
            self.last_pos_msg = deepcopy(msg)
            trans = self.grid.trans(diff.dx, diff.dy)
            rot = self.grid.rot(diff.dpsi)

            if trans or rot:
                self.plot_updated = True
            self.pos_lock.release()

    def update_plot(self):
        if self.plot_updated:
            if Settings.plot_type == 0:
                self.img_item.setImage(self.grid.get_raw(), levels=(0.0, 255.0), autoLevels=False)
            elif Settings.plot_type == 1:
                if self.binary_plot_on:
                    self.img_item.setImage(self.grid.get_binary_map(), levels=(0, 1), autoLevels=False)
                else:
                    self.img_item.setImage(self.grid.get_p(), levels=(-5.0, 5.0), autoLevels=False)
            elif Settings.plot_type == 2:
                im, self.map_obstacle_list = self.grid.adaptive_threshold(self.threshold_box.value())
                self.img_item.setImage(im)
            else:
                raise Exception('Invalid plot type')
            self.img_item.setPos(-self.grid.last_distance,
                                 -self.grid.last_distance/2 if GridSettings.half_grid else - self.grid.last_distance)
            # self.img_item.scale(self.grid.last_distance/self.grid.RES, self.grid.last_distance/self.grid.RES)

            if Settings.hist_window:
                hist, _ = np.histogram(self.grid.get_raw(), 256)
                print(np.sum(hist[1:]*np.arange(1, 256)/np.sum(hist[1:])))
                self.histogram.setOpts(x=np.arange(255), y1=hist[1:], height=1000)
                self.hist_window.setYRange(max=1000, min=0)

            if Settings.collision_avoidance and Settings.show_map:
                self.map_view.repaint()
            self.plot_updated = False

    def collision_avoidance_loop(self):
        if Settings.input_source == 1:
            self.moos_msg_client.send_msg('get_waypoints', 0)
        else:
            raise NotImplemented()

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
