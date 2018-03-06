import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
import threading
from copy import deepcopy

from settings import *
from ogrid.oGrid import OGrid
from messages.UdpMessageClient import UdpMessageClient
from messages.moosMsgs import MoosMsgs
from messages.moosPosMsg import *
from collision_avoidance.collisionAvoidance import CollisionAvoidance
import cv2
import map

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
    contour_list = []

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        if Settings.input_source == 0:
            raise NotImplemented
            self.last_pos_msg = None
        elif Settings.input_source == 1:
            self.last_pos_msg = MoosPosMsg()
            self.last_pos_diff = MoosPosMsgDiff(0, 0, 0)

        main_layout = QtGui.QHBoxLayout()  # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QGridLayout()

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

        # Adding items
        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.binary_plot_button)
        left_layout.addWidget(self.clear_grid_button)

        right_layout.addWidget(graphics_view, 0, 0, 1, 2)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        if Settings.collision_avoidance and Settings.show_map:
            self.map_widget = map.MapWidget()
            self.map_widget.setMaximumSize(800, 10**6)
            right_layout.addWidget(self.map_widget, 0, 2, 1, 1)
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
                self.moos_msg_client.signal_new_wp.connect(self.wp_received)
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
                raise NotImplemented
            else:
                msg = self.moos_msg_client.cur_pos_msg
            if self.last_pos_msg is None:
                self.last_pos_msg = deepcopy(msg)

            if Settings.collision_avoidance:
                self.collision_avoidance.update_pos(msg.lat, msg.long, msg.psi)
                if Settings.show_map:
                    self.map_widget.update_pos(msg.lat, msg.long, msg.psi, self.grid.range_scale)
                    self.map_widget.update_waypoints(self.collision_avoidance.waypoint_list,
                                                     self.collision_avoidance.waypoint_counter)
                    self.map_widget.update_avoidance_waypoints(self.collision_avoidance.new_wp_list)

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
                im, ellipses, contours = self.grid.adaptive_threshold(self.threshold_box.value())
                self.collision_avoidance.update_obstacles(contours, self.grid.range_scale)
                if self.collision_avoidance.voronoi_wp_list is not None and len(self.collision_avoidance.voronoi_wp_list) > 0:
                    for wp0, wp1 in zip(self.collision_avoidance.voronoi_wp_list, self.collision_avoidance.voronoi_wp_list[1:]):
                        cv2.line(im, wp0, wp1, (255, 0, 0), 2)
                if Settings.show_map:
                    self.map_widget.update_obstacles(contours, self.grid.range_scale, self.last_pos_msg.lat,
                                                     self.last_pos_msg.long, self.last_pos_msg.psi)
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

            # if Settings.collision_avoidance and Settings.show_map:
            #     self.map_widget.repaint()
            self.plot_updated = False

    def collision_avoidance_loop(self):
        if Settings.input_source == 1:
            self.moos_msg_client.send_msg('get_waypoints', 0)
        else:
            raise NotImplemented()
        # self.collision_avoidance.loop()

    @QtCore.pyqtSlot(list, int)
    def wp_received(self, waypoint_list, waypoint_counter):
        print('signal received')
        self.collision_avoidance.callback(waypoint_list, waypoint_counter)

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
