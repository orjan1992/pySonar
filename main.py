import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
import threading
from copy import deepcopy
import sys
from time import strftime

from settings import *
from ogrid.rawGrid import RawGrid
from ogrid.occupancyGrid import OccupancyGrid
from messages.UdpMessageClient import UdpMessageClient
from messages.moosMsgs import MoosMsgs
from messages.moosPosMsg import *
from collision_avoidance.collisionAvoidance import CollisionAvoidance
import map

# LOG and EXECPTION stuff
LOG_FILENAME = 'main.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)
logger = logging.getLogger('main')
logging.getLogger('messages.MoosMsgs.pose').disabled = True
logging.getLogger('messages.MoosMsgs.bins').disabled = True
logging.getLogger('messages.MoosMsgs.pose').disabled = True
logging.getLogger('moosPosMsg').disabled = True
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    return

# Main Prog
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
    collision_stat = 0
    thread_pool = QtCore.QThreadPool()

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
            colormap = pg.ColorMap(PlotSettings.steps_raw, np.array(
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

        # Collision margin box
        self.collision_margin_box = QtGui.QDoubleSpinBox()
        self.collision_margin_box.setMinimum(0)
        self.collision_margin_box.setValue(CollisionSettings.obstacle_margin)
        self.collision_margin_box.setSingleStep(0.5)
        self.collision_margin_box.valueChanged.connect(self.update_collision_margin)

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
        left_layout.addWidget(self.collision_margin_box)
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

        if Settings.save_scan_lines:
            self.scan_lines = []

        if Settings.hist_window:
            self.hist_window = pg.PlotWindow()
            self.histogram = pg.BarGraphItem(x=np.arange(10), y1=np.random.rand(10), width=0.3, brush='r')
            self.hist_window.addItem(self.histogram)
            self.hist_window.show()
        if Settings.show_voronoi_plot:
            self.voronoi_window = pg.PlotWindow()
            self.voronoi_plot_item = pg.ImageItem(autoLevels=False)
            self.voronoi_window.addItem(self.voronoi_plot_item)
            self.voronoi_plot_item.getViewBox().invertY(True)
            self.voronoi_plot_item.getViewBox().setAspectLocked(True)
            self.voronoi_plot_item.setOpts(axisOrder='row-major')
            self.voronoi_window.show()

        self.init_grid()
        if Settings.input_source == 0:
            self.udp_client = UdpMessageClient(ConnectionSettings.sonar_port, self.new_sonar_msg)
            client_thread = threading.Thread(target=self.udp_client.connect, daemon=True)
            client_thread.start()
        elif Settings.input_source == 1:
            self.moos_msg_client = MoosMsgs(self.new_sonar_msg)
            self.moos_msg_client.signal_new_sonar_msg.connect(self.new_sonar_msg)
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
                if Settings.show_voronoi_plot:
                    self.collision_avoidance = CollisionAvoidance(self.moos_msg_client, self.voronoi_plot_item)
                else:
                    self.collision_avoidance = CollisionAvoidance(self.moos_msg_client)
                self.moos_msg_client.signal_new_wp.connect(self.wp_received)
            else:
                raise NotImplemented
            self.collision_worker = CollisionAvoidanceWorker(self.collision_avoidance)
            self.collision_worker.setAutoDelete(False)
            self.collision_worker.signals.finished.connect(self.collision_loop_finished)

            self.collision_avoidance_timer = QtCore.QTimer()
            self.collision_avoidance_timer.setSingleShot(True)
            self.collision_avoidance_timer.timeout.connect(self.collision_avoidance_loop)
            self.collision_avoidance_timer.start(Settings.collision_avoidance_interval)

        self.pos_update_timer.start(Settings.pos_update)

    def init_grid(self):
        if Settings.update_type == 1:
            self.grid = OccupancyGrid(GridSettings.half_grid, GridSettings.p_inital, GridSettings.p_occ,
                                      GridSettings.p_free, GridSettings.p_binary_threshold, GridSettings.cell_factor)
        elif Settings.update_type == 0:
            self.grid = RawGrid(GridSettings.half_grid)

    def clear_grid(self):
        self.grid.clear_grid()
        self.plot_updated = False
        self.update_plot()

    @QtCore.pyqtSlot(object, name='new_sonar_msg')
    def new_sonar_msg(self, msg):
        self.grid.update_distance(msg.range_scale)
        if Settings.update_type == 0:
            self.grid.update_raw(msg)
        elif Settings.update_type == 1:
            self.grid.auto_update_zhou(msg, self.threshold_box.value())
        else:
            raise Exception('Invalid update type')
        self.plot_updated = True
        if Settings.save_scan_lines:
            self.scan_lines.append(msg.data)
            if len(self.scan_lines) > 100:
                np.savez('pySonarLog/scan_lines_{}'.format(strftime("%Y%m%d-%H%M%S")),
                         scan_lines=np.array(self.scan_lines))
                self.scan_lines = []


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
                    # self.map_widget.update_avoidance_waypoints(self.collision_avoidance.new_wp_list)

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
                    self.img_item.setImage(self.grid.get_p(), levels=(0.0, 1.0), autoLevels=False)
            elif Settings.plot_type == 2:
                if Settings.update_type == 1:
                    im, contours = self.grid.get_obstacles()
                else:
                    im, contours = self.grid.adaptive_threshold(self.threshold_box.value())
                self.collision_avoidance.update_obstacles(contours, self.grid.range_scale)

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
        # TODO: faster loop when no object is in the way
        self.collision_worker.set_reliable(self.grid.reliable)
        self.thread_pool.start(self.collision_worker)


    @QtCore.pyqtSlot(int, name='collision_worker_finished')
    def collision_loop_finished(self, status):
        self.collision_stat = status
        if self.collision_stat == 2:
            self.map_widget.invalidate_wps()
            self.collision_avoidance_timer.start(0)
        else:
            self.collision_avoidance_timer.start(Settings.collision_avoidance_interval)

    @QtCore.pyqtSlot(object, name='new_wp')
    def wp_received(self, var):
        if type(var) is list:
            self.collision_avoidance.update_external_wps(var, None)
        elif type(var) is int:
            self.collision_avoidance.update_external_wps(None, var)
        else:
            raise Exception('Unknown object type in wp_received slot')
        if Settings.show_map:
            self.map_widget.update_waypoints(self.collision_avoidance.waypoint_list,
                                             self.collision_avoidance.waypoint_counter, self.collision_stat)


    def binary_button_click(self):
        if self.binary_plot_on:
            self.binary_plot_button.text = "Set Prob mode"
        else:
            self.binary_plot_button.text = "Set Binary mode"
        self.binary_plot_on = not self.binary_plot_on

    def update_collision_margin(self):
        CollisionSettings.obstacle_margin = self.collision_margin_box.value()


class CollisionAvoidanceWorker(QtCore.QRunnable):

    def __init__(self, collision_avoidance):
        super().__init__()
        self.collision_avoidance = collision_avoidance
        self.reliable = False
        self.signals = CollisionAvoidanceWorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        status = self.collision_avoidance.main_loop(self.reliable)
        self.signals.finished.emit(status)

    def set_reliable(self, reliable):
        self.reliable = reliable


class CollisionAvoidanceWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, name='collision_worker_finished')


if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

    window.login_widget.collision_avoidance.save_paths()
    if Settings.save_scan_lines:
        np.savez('pySonarLog/scan_lines_{}'.format(strftime("%Y%m%d-%H%M%S")), scan_lines=np.array(window.login_widget.scan_lines))
