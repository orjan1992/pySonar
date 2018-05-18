import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
import threading
from copy import deepcopy
import sys
from time import strftime
from coordinate_transformations import vehicle2NED
from settings import *
from ogrid.rawGrid import RawGrid
from ogrid.occupancyGrid import OccupancyGrid
from messages.AutoPilotMsg import RemoteControlRequest
if Settings.input_source == 0:
    from messages.udpClient_py import UdpClient
elif Settings.input_source == 1:
    from messages.moosMsgs import MoosMsgs
from messages.moosPosMsg import *
from collision_avoidance.collisionAvoidance import CollisionAvoidance, CollisionStatus
import map
from messages.udpMsg import *
from scipy.io import savemat
from time import strftime, time
import messages.AutoPilotMsg as ap
from collision_avoidance.los_controller import LosController

# LOG and EXECPTION stuff
LOG_FILENAME = 'main.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')
logging.getLogger('messages.MoosMsgs.pose').disabled = True
logging.getLogger('messages.MoosMsgs.bins').disabled = True
logging.getLogger('messages.MoosMsgs.pose').disabled = True
# logging.getLogger('Collision_avoidance').disabled = True
logging.getLogger('moosPosMsg').disabled = True
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
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
        self.setWindowTitle('pySonar')


# noinspection PyUnresolvedReferences
class MainWidget(QtGui.QWidget):
    plot_updated = False
    grid = None
    contour_list = []
    collision_stat = 0
    thread_pool = QtCore.QThreadPool()

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        # if Settings.input_source == 0:
        #     raise NotImplemented
        #     self.last_pos_msg = None
        # elif Settings.input_source == 1:
        #     self.last_pos_msg = MoosPosMsg()
        #     self.last_pos_diff = MoosPosMsgDiff(0, 0, 0)
        self.last_pos_msg = MoosPosMsg()
        self.last_pos_diff = MoosPosMsgDiff(0, 0, 0)

        main_layout = QtGui.QHBoxLayout()  # Main layout
        left_layout = QtGui.QVBoxLayout()
        right_layout = QtGui.QVBoxLayout()

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
        self.threshold_box.setMaximum(20000)
        self.threshold_box.setValue(GridSettings.threshold)

        # Randomize button
        self.randomize_button = QtGui.QPushButton('Randomize')
        self.randomize_button.clicked.connect(self.randomize_occ_grid)
        # wp straight ahead button
        self.wp_straight_ahead_button = QtGui.QPushButton('Wp Straight Ahed')
        self.wp_straight_ahead_button.clicked.connect(self.wp_straight_ahead_clicked)

        # Collision margin box
        self.collision_margin_box = QtGui.QDoubleSpinBox()
        self.collision_margin_box.setMinimum(0)
        self.collision_margin_box.setValue(CollisionSettings.obstacle_margin)
        self.collision_margin_box.setSingleStep(0.5)
        self.collision_margin_box.valueChanged.connect(self.update_collision_margin)

        # binary plot
        self.binary_plot_button = QtGui.QPushButton('Send Initial WPs')
        self.binary_plot_button.clicked.connect(self.binary_button_click)

        # Clear grid button
        self.clear_grid_button = QtGui.QPushButton('Clear Grid!')
        self.clear_grid_button.clicked.connect(self.clear_grid)

        # Adding items
        self.threshold_box.setMaximumSize(Settings.button_width, Settings.button_height)
        self.collision_margin_box.setMaximumSize(Settings.button_width, Settings.button_height)
        self.binary_plot_button.setMaximumSize(Settings.button_width, Settings.button_height)
        self.clear_grid_button.setMaximumSize(Settings.button_width, Settings.button_height)
        self.randomize_button.setMaximumSize(Settings.button_width, Settings.button_height)
        self.wp_straight_ahead_button.setMaximumSize(Settings.button_width, Settings.button_height)

        left_layout.addWidget(self.threshold_box)
        left_layout.addWidget(self.collision_margin_box)
        left_layout.addWidget(self.binary_plot_button)
        left_layout.addWidget(self.clear_grid_button)
        left_layout.addWidget(self.randomize_button)
        left_layout.addWidget(self.wp_straight_ahead_button)
        # left_layout.setGeometry(QtCore.QRect(0, 0, 200, 10**6))
        # left_layout.SizeHint(QtCore.QRect(0, 0,))
        # print(left_layout.maximumSize(200, 0))
        right_layout.addWidget(graphics_view)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        if Settings.collision_avoidance and Settings.show_map:
            self.map_widget = map.MapWidget()
            self.map_widget.setMaximumSize(800, 10**6)
            main_layout.addWidget(self.map_widget)
        self.setLayout(main_layout)

        if Settings.show_pos:
            pos_layout = QtGui.QVBoxLayout()
            text_layout = QtGui.QHBoxLayout()
            value_layout = QtGui.QHBoxLayout()
            N_text = QtGui.QLabel('North')
            N_text.setLineWidth(3)
            E_text = QtGui.QLabel('East')
            E_text.setLineWidth(3)
            H_text = QtGui.QLabel('Heading')
            H_text.setLineWidth(3)
            self.north = QtGui.QLabel('0')
            self.north.setLineWidth(3)
            self.east = QtGui.QLabel('0')
            self.east.setLineWidth(3)
            self.heading = QtGui.QLabel('0')
            self.heading.setLineWidth(3)
            text_layout.addWidget(N_text)
            text_layout.addWidget(E_text)
            text_layout.addWidget(H_text)
            value_layout.addWidget(self.north)
            value_layout.addWidget(self.east)
            value_layout.addWidget(self.heading)
            pos_layout.addLayout(text_layout)
            pos_layout.addLayout(value_layout)
            right_layout.addLayout(pos_layout)

            if LosSettings.enable_los:
                cross_track_text = QtGui.QLabel('Cross-track error')
                cross_track_text.setLineWidth(3)
                along_track_text = QtGui.QLabel('Along-track error')
                along_track_text.setLineWidth(3)
                self.cross_track = QtGui.QLabel('0')
                self.cross_track.setLineWidth(3)
                self.along_track = QtGui.QLabel('0')
                self.along_track.setLineWidth(3)
                text_layout.addWidget(cross_track_text)
                text_layout.addWidget(along_track_text)
                value_layout.addWidget(self.cross_track)
                value_layout.addWidget(self.along_track)

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
            self.udp_client = UdpClient(ConnectionSettings.sonar_port, ConnectionSettings.pos_port,
                                        ConnectionSettings.autopilot_ip, ConnectionSettings.autopilot_server_port,
                                        ConnectionSettings.autopilot_listen_port)
            self.udp_client.signal_new_sonar_msg.connect(self.new_sonar_msg)
            self.udp_client.set_sonar_callback(self.new_sonar_msg)
            self.udp_client.start()
            # self.udp_thread = threading.Thread(target=self.udp_client.start)
            # self.udp_thread.setDaemon(True)
            # self.udp_thread.start()
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
                if Settings.show_voronoi_plot:
                    self.collision_avoidance = CollisionAvoidance(self.udp_client, self.voronoi_plot_item)
                else:
                    self.collision_avoidance = CollisionAvoidance(self.udp_client)
            self.collision_worker = CollisionAvoidanceWorker(self.collision_avoidance)
            self.collision_worker.setAutoDelete(False)
            self.collision_worker.signals.finished.connect(self.collision_loop_finished)

            self.collision_avoidance_timer = QtCore.QTimer()
            self.collision_avoidance_timer.setSingleShot(True)
            self.collision_avoidance_timer.timeout.connect(self.collision_avoidance_loop)
            self.collision_avoidance_timer.start(Settings.collision_avoidance_interval)

        self.grid_worker = GridWorker(self.grid)
        self.grid_worker.setAutoDelete(False)
        self.grid_worker.signals.finished.connect(self.grid_worker_finished)

        self.pos_update_timer.start(Settings.pos_update_speed)
        self.grid_worker_finished(True)


    def init_grid(self):
        if Settings.update_type == 1:
            self.grid = OccupancyGrid(GridSettings.half_grid, GridSettings.p_inital, GridSettings.p_occ,
                                      GridSettings.p_free, GridSettings.p_binary_threshold, GridSettings.cell_factor)
        elif Settings.update_type == 0:
            self.grid = RawGrid(GridSettings.half_grid)

    def clear_grid(self):
        # self.grid.clear_grid()
        # self.plot_updated = False
        # self.update_plot()
        self.grid_worker.clear_grid()

    @QtCore.pyqtSlot(object, name='new_sonar_msg')
    def new_sonar_msg(self, msg):
        self.grid_worker.add_data_msg(msg, self.threshold_box.value())
        self.plot_updated = True
        if Settings.save_scan_lines:
            self.scan_lines.append(msg.data)
            if len(self.scan_lines) > 100:
                np.savez('pySonarLog/scan_lines_{}'.format(strftime("%Y%m%d-%H%M%S")),
                         scan_lines=np.array(self.scan_lines))
                self.scan_lines = []


    def new_pos_msg(self):
        with self.pos_lock:
            if Settings.input_source == 0:
                msg = self.udp_client.cur_pos_msg
                if msg is None:
                    return
            else:
                msg = self.moos_msg_client.cur_pos_msg
            if Settings.show_pos:
                self.north.setText('{:.2f}'.format(msg.north))
                self.east.setText('{:.2f}'.format(msg.east))
                self.heading.setText('{:.1f}'.format(msg.yaw*180.0/np.pi))
                if LosSettings.enable_los and Settings.enable_autopilot:
                    e, s = self.udp_client.los_controller.get_errors()
                    self.collision_avoidance.update_external_wps(wp_counter=self.udp_client.los_controller.get_wp_counter())
                    self.along_track.setText('{:.2f}'.format(s))
                    self.cross_track.setText('{:.2f}'.format(e))
            # if self.last_pos_msg is None:
            #     self.last_pos_msg = deepcopy(msg)

            if Settings.collision_avoidance:
                self.collision_avoidance.update_pos(msg)
                if Settings.show_map:
                    self.map_widget.update_pos(msg.north, msg.east, msg.yaw, self.grid.range_scale)
                    # self.map_widget.update_avoidance_waypoints(self.collision_avoidance.new_wp_list)

            self.grid_worker.update(msg)
            self.last_pos_msg = deepcopy(msg)

            # diff = (msg - self.last_pos_msg)
            # trans = self.grid.trans(diff.dx, diff.dy)
            # rot = self.grid.rot(diff.dyaw)
            # if trans or rot:
            #     self.plot_updated = True

            # if self.thread_pool.activeThreadCount() < self.thread_pool.maxThreadCount():
            #     diff = (msg - self.last_pos_msg)
            #     self.last_pos_msg = deepcopy(msg)
            #
            #     self.thread_pool.start(self.grid_worker, 6)
            #     logger.debug('Start grid worker: {} of {}'.format(self.thread_pool.activeThreadCount(), self.thread_pool.maxThreadCount()))
            # else:
            #     logger.debug('Skipped iteration because of few available threads')

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
                    self.img_item.setImage(np.zeros((1601, 1601, 3)))
                    image, contours = self.grid.get_obstacles()
                    self.img_item.setImage(image)
                    a=1
                else:
                    im, contours = self.grid.adaptive_threshold(self.threshold_box.value())
                if Settings.collision_avoidance:
                    self.collision_avoidance.update_obstacles(contours, self.grid.range_scale)
                    if Settings.show_wp_on_grid:
                        if self.last_pos_msg is None:
                            im = self.collision_avoidance.draw_wps_on_grid(image, (0,0,0))
                        else:
                            im = self.collision_avoidance.draw_wps_on_grid(image, (self.last_pos_msg.north, self.last_pos_msg.east, self.last_pos_msg.yaw))

                if Settings.show_map:
                    self.map_widget.update_obstacles(contours, self.grid.range_scale, self.last_pos_msg.north,
                                                     self.last_pos_msg.east, self.last_pos_msg.yaw)
                self.img_item.setImage(im)
            else:
                raise Exception('Invalid plot type')
            if self.grid.last_distance is not None:
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
        self.collision_worker.set_reliable(self.grid.reliable)
        self.thread_pool.start(self.collision_worker, 6)
        # logger.debug('Start collision worker: {} of {}'.format(self.thread_pool.activeThreadCount(), self.thread_pool.maxThreadCount()))

    @QtCore.pyqtSlot(CollisionStatus, name='collision_worker_finished')
    def collision_loop_finished(self, status):
        self.collision_stat = status
        if status is CollisionStatus.NO_FEASIBLE_ROUTE or status is CollisionStatus.SMOOTH_PATH_VIOLATES_MARGIN:
            left = np.mean(self.grid.grid[:, :800])
            right = np.mean(self.grid.grid[:, 801:])
            if Settings.input_source == 0:
                self.udp_client.stop_and_turn(np.sign(left-right) * 15* np.pi / 180.0)
            elif Settings.show_map:
                self.map_widget.invalidate_wps()
            if Settings.show_map:
                self.map_widget.invalidate_wps()
            self.collision_avoidance_timer.start(0)
            return
        elif status is CollisionStatus.NEW_ROUTE_OK:
            if Settings.show_wp_on_grid:
                self.plot_updated = True
        if Settings.save_obstacles and status is not CollisionStatus.NO_DANGER:
            self.grid_worker.save_obs()
        self.collision_avoidance_timer.start(Settings.collision_avoidance_interval)

    @QtCore.pyqtSlot(bool, name='grid_worker_finished')
    def grid_worker_finished(self, status):
        if status:
            self.plot_updated = True
        self.thread_pool.start(self.grid_worker, 6)
        # logger.debug('Start grid worker: {} of {}'.format(self.thread_pool.activeThreadCount(), self.thread_pool.maxThreadCount()))

    @QtCore.pyqtSlot(object, name='new_wp')
    def wp_received(self, var):
        if type(var) is list:
            self.collision_avoidance.update_external_wps(var, None)
        elif type(var) is int:
            self.collision_avoidance.update_external_wps(None, var)
        else:
            raise Exception('Unknown object type in wp_received slot')
        if Settings.show_map:
            wp_list, wp_counter = self.collision_avoidance.data_storage.get_wps()
            self.map_widget.update_waypoints(wp_list, wp_counter, self.collision_stat)


    def binary_button_click(self):
        if Settings.collision_avoidance:
            wp = np.load('collision_avoidance/smooth_wgs84.npz')['smooth']
            # wp_list = np.ndarray.tolist(wp)
            # wp_list = [[6821587.4301, 457961.291, 4],
            #     [6821573.0927, 457944.6148, 4],
            #     [6821563.7182, 457947.2479, 4],
            #     [6821563.1668, 457959.5356, 4],
            #     [6821574.1955, 457981.478, 4],
            #     [6821553.2409, 457994.6434, 4],
            #     [6821521.2575, 457996.8376, 4],
            #     [6821493.1341, 457993.7657, 4],
            #     [6821482.1053, 458013.5139, 4]]
            # wp_list.pop(0)
            # wp_list.pop(0)

            wp_list = [[6821592.4229, 457959.8588, 3],
                       [6821574.0057, 457960.6302, 3],
                       [6821542.018, 457939.8021, 3],
                       [6821573.5211, 457922.831, 3],
                       [6821570.6131, 458066.3137, 3],
                       [6821582.245, 457967.9586, 3],
                       [6821593.8769, 457920.1311, 3],
                       [6821621.5027, 457966.0301, 3],
                       [6821545.8953, 458010.772, 3]]

            wp0 = vehicle2NED(0, 0, self.last_pos_msg.north, self.last_pos_msg.east, self.last_pos_msg.yaw)
            wp_list.insert(0, [wp0[0], wp0[1], 2.0])
            self.collision_avoidance.update_external_wps(wp_list, 0)
            self.collision_avoidance.save_paths(wp_list)
            self.udp_client.update_wps(wp_list)
            self.plot_updated = True

    def wp_straight_ahead_clicked(self):
        if Settings.collision_avoidance == True:
            if self.grid.range_scale == 1:
                self.grid.range_scale = 30
            with self.pos_lock:
                if self.last_pos_msg is None:
                    self.last_pos_msg = MoosPosMsg(6821592.4229, 457959.8588, 0, 2)
                    self.collision_avoidance.update_pos(self.last_pos_msg)
                wp1 = vehicle2NED(self.grid.range_scale*CollisionSettings.dummy_wp_factor[0],
                                  self.grid.range_scale * CollisionSettings.dummy_wp_factor[1], self.last_pos_msg.north,
                                 self.last_pos_msg.east, self.last_pos_msg.yaw)
                wp0 = vehicle2NED(1, 0, self.last_pos_msg.north, self.last_pos_msg.east, self.last_pos_msg.yaw)
                wp0 = [wp0[0], wp0[1], 140, 0.5]

            wp1 = [wp1[0], wp1[1], 140, 0.5]
            self.collision_avoidance.update_external_wps([wp0, wp1], 0)
            self.udp_client.update_wps([wp0, wp1])
            self.plot_updated = True

    def randomize_occ_grid(self):
        if Settings.update_type == 1:
            # self.grid.randomize()
            # self.plot_updated = True
            self.grid_worker.randomize()
        # if Settings.collision_avoidance == True:
        #     # self.collision_avoidance.update_obstacles(self.grid.get_obstacles()[1], self.grid.range_scale)
        #     self.wp_straight_ahead_clicked()
        # else:
        self.plot_updated = True

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
    finished = QtCore.pyqtSignal(CollisionStatus, name='collision_worker_finished')


class GridWorker(QtCore.QRunnable):
    def __init__(self, grid, diff=MoosPosMsgDiff(0, 0, 0)):
        super().__init__()
        self.grid = grid
        self.signals = GridWorkerSignals()
        self.diff = diff
        self.random = False
        self.clear_grid_bool = False
        self.lock = threading.Lock()
        self.data_list = []
        self.threshold = GridSettings.threshold
        self.pos = MoosPosMsg(0, 0, 0)
        self.last_pos = None
        # self.runtime = np.zeros(50)
        # self.runtime_counter = 0
        # self.dx = 0

        if Settings.save_obstacles:
            self.save_obs_counter = 0
            self.save_obs_timer = QtCore.QTimer()
            self.save_obs_timer.timeout.connect(self.save_obs)
            self.save_obs_timer.setSingleShot(False)
            self.save_obs_timer.start(10000)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            with self.lock:
                # diff = self.diff
                msg_list = self.data_list.copy()
                self.data_list.clear()
                threshold = self.threshold
                random = self.random
                if self.random:
                    self.random = False
                clear_grid = self.clear_grid_bool
                if self.clear_grid_bool:
                    self.clear_grid_bool = False
                # if Settings.save_obstacles:
                pos = self.pos
            if random:
                self.grid.randomize()
                self.grid.calc_obstacles()
            elif clear_grid:
                self.grid.clear_grid()
                self.grid.calc_obstacles()
            else:
                t1 = time()
                if self.last_pos is None:
                    if pos.north != 0 and pos.east != 0:
                        self.last_pos = pos
                else:
                    diff = pos - self.last_pos
                    # self.dx += diff.dx
                    # logger.debug(str(diff))
                    self.last_pos = pos
                    self.grid.trans_and_rot(diff)
                if Settings.plot_type == 0:
                    for msg in msg_list:
                        self.grid.update_distance(msg.range_scale)

                        self.grid.update_raw(msg)
                        self.grid.calc_obstacles(threshold)
                elif Settings.plot_type == 2:
                    # self.grid.auto_update_zhou(msg, self.threshold_box.value())
                    if len(msg_list) == 1:
                        self.grid.update_distance(msg_list[0].range_scale)
                        self.grid.update_occ_zhou(msg_list[0], threshold)
                        self.grid.calc_obstacles()
                    elif len(msg_list) > 1:
                        self.grid.update_distance(msg_list[0].range_scale)
                        grid = self.grid.update_occ_zhou(msg_list[0], threshold, multi_update=True)
                        if len(msg_list) > 2:
                            for i in range(1, len(msg_list)-1):
                                self.grid.update_distance(msg_list[i].range_scale)
                                grid = self.grid.update_occ_zhou(msg_list[i], threshold, multi_update=True, multigrid=grid)
                        self.grid.update_distance(msg_list[-1].range_scale)
                        self.grid.update_occ_zhou(msg_list[-1], threshold, multi_update=False, multigrid=grid)
                    else:
                        pass
                else:
                    raise Exception('Invalid update type')
                # self.runtime[self.runtime_counter] = (time()-t1)*1000
                # self.runtime_counter += 1
                # if self.runtime_counter == 50:
                #     self.runtime_counter = 0
                # logger.info('Grid loop time: {:3.2f} ms\tlist_len: {}'.format(np.mean(self.runtime), len(msg_list)))
                # logger.info(len(msg_list))
            self.signals.finished.emit(True)
        except Exception as e:
            logger.error('Grid Worker', e)

    def save_obs(self):
        with self.lock:
            pos = self.pos
        if self.grid.contours is None:
            c = []
        else:
            c = self.grid.contours
        if pos.north != 0 and pos.east != 0:
            savemat('C:/Users/Ørjan/Desktop/logs/obstacles{}'.format(strftime("%Y%m%d-%H%M%S")),
                    mdict={'grid': self.grid.grid.astype(np.float16), 'obs': c,
                           'pos': np.array([pos.north, pos.east, pos.yaw]), 'range_scale': self.grid.range_scale})

    def update(self, pos):
        with self.lock:
            self.pos = pos

    def randomize(self):
        with self.lock:
            self.random = True

    def add_data_msg(self, msg, threshold):
        with self.lock:
            self.data_list.append(msg)
            self.threshold = threshold

    def clear_grid(self):
        with self.lock:
            self.clear_grid_bool = True

class GridWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(bool, name='grid_worker_finished')


if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
    # print(window.login_widget.grid_worker.dx)
    if Settings.collision_avoidance:
        window.login_widget.collision_avoidance.save_paths()
    if Settings.show_voronoi_plot:
        window.login_widget.voronoi_window.close()
    if Settings.save_scan_lines:
        np.savez('pySonarLog/scan_lines_{}'.format(strftime("%Y%m%d-%H%M%S")), scan_lines=np.array(window.login_widget.scan_lines))
    if Settings.input_source == 0:
        window.login_widget.udp_client.close()
