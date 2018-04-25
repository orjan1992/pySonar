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
import messages.AutoPilotMsg as ap
from collision_avoidance.los_controller import LosController

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
        self.threshold_box.setValue(PlotSettings.threshold)

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
        self.grid.clear_grid()
        self.plot_updated = False
        self.update_plot()

    @QtCore.pyqtSlot(object, name='new_sonar_msg')
    def new_sonar_msg(self, msg):
        self.grid.update_distance(msg.range_scale)
        if Settings.update_type == 0:
            self.grid.update_raw(msg)
        elif Settings.update_type == 1:
            # self.grid.auto_update_zhou(msg, self.threshold_box.value())
            self.grid.update_occ_zhou(msg, self.threshold_box.value())
            # self.grid.new_occ_update(msg, self.threshold_box.value())
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
        if self.pos_lock.acquire():
            if Settings.input_source == 0:
                msg = self.udp_client.cur_pos_msg
                if msg is None:
                    self.pos_lock.release()
                    return
            else:
                msg = self.moos_msg_client.cur_pos_msg
            if Settings.show_pos:
                self.north.setText('{:.6f}'.format(msg.lat))
                self.east.setText('{:.6f}'.format(msg.long))
                self.heading.setText('{:.1f}'.format(msg.psi*180.0/np.pi))
                if LosSettings.enable_los:
                    e, s = self.udp_client.los_controller.get_errors()
                    self.along_track.setText('{:.2f}'.format(s))
                    self.cross_track.setText('{:.2f}'.format(e))
            if self.last_pos_msg is None:
                self.last_pos_msg = deepcopy(msg)

            if Settings.collision_avoidance:
                self.collision_avoidance.update_pos(msg.lat, msg.long, msg.psi)
                if Settings.show_map:
                    self.map_widget.update_pos(msg.lat, msg.long, msg.psi, self.grid.range_scale)
                    # self.map_widget.update_avoidance_waypoints(self.collision_avoidance.new_wp_list)

            diff = (msg - self.last_pos_msg)
            self.last_pos_msg = deepcopy(msg)
            # trans = self.grid.trans(diff.dx, diff.dy)
            # rot = self.grid.rot(diff.dpsi)
            # if trans or rot:
            #     self.plot_updated = True

            # if self.thread_pool.activeThreadCount() < self.thread_pool.maxThreadCount():
            #     diff = (msg - self.last_pos_msg)
            #     self.last_pos_msg = deepcopy(msg)
            #
            self.grid_worker.update(diff)
            #     self.thread_pool.start(self.grid_worker, 6)
            #     logger.debug('Start grid worker: {} of {}'.format(self.thread_pool.activeThreadCount(), self.thread_pool.maxThreadCount()))
            # else:
            #     logger.debug('Skipped iteration because of few available threads')
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
                if Settings.collision_avoidance:
                    self.collision_avoidance.update_obstacles(contours, self.grid.range_scale)
                    if Settings.show_wp_on_grid:
                        if self.last_pos_msg is None:
                            im = self.collision_avoidance.draw_wps_on_grid(im, (0,0,0))
                        else:
                            im = self.collision_avoidance.draw_wps_on_grid(im, (self.last_pos_msg.lat, self.last_pos_msg.long, self.last_pos_msg.psi))

                if Settings.show_map:
                    self.map_widget.update_obstacles(contours, self.grid.range_scale, self.last_pos_msg.lat,
                                                     self.last_pos_msg.long, self.last_pos_msg.psi)
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
                self.udp_client.send_autopilot_msg(ap.GuidanceMode(ap.GuidanceModeOptions.STATION_KEEPING))
                self.udp_client.send_autopilot_msg(ap.Setpoint(np.sign(left-right) * np.pi / 2, ap.Dofs.YAW, False))
            elif Settings.show_map:
                self.map_widget.invalidate_wps()
            if Settings.show_map:
                self.map_widget.invalidate_wps()
            self.collision_avoidance_timer.start(0)
            return
        elif status is CollisionStatus.NEW_ROUTE_OK:
            if Settings.show_wp_on_grid:
                self.plot_updated = True
        self.collision_avoidance_timer.start(Settings.collision_avoidance_interval)

    @QtCore.pyqtSlot(bool, name='grid_worker_finished')
    def grid_worker_finished(self, status):
        if status:
            self.plot_updated = True
        self.thread_pool.start(self.grid_worker, 6)

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
        if self.binary_plot_on:
            self.binary_plot_button.text = "Set Prob mode"
        else:
            self.binary_plot_button.text = "Set Binary mode"
        self.binary_plot_on = not self.binary_plot_on

    def wp_straight_ahead_clicked(self):
        if Settings.collision_avoidance == True:
            if self.grid.range_scale == 1:
                self.grid.range_scale = 30
            self.pos_lock.acquire()
            if self.last_pos_msg is None:
                self.last_pos_msg = MoosPosMsg(0, 0, 0, 0)
            wp1 = vehicle2NED(self.grid.range_scale*CollisionSettings.dummy_wp_factor[0],
                              self.grid.range_scale * CollisionSettings.dummy_wp_factor[1], self.last_pos_msg.lat,
                             self.last_pos_msg.long, self.last_pos_msg.psi)
            wp0 = vehicle2NED(2, 0, self.last_pos_msg.lat, self.last_pos_msg.long, self.last_pos_msg.psi)
            wp0 = [wp0[0], wp0[1], 0, 0.5]
            self.pos_lock.release()
            wp1 = [wp1[0], wp1[1], 0, 0.5]
            self.collision_avoidance.update_external_wps([wp0, wp1], 0)
            self.udp_client.update_wps([wp0, wp1])
            self.plot_updated = True

    def randomize_occ_grid(self):
        if Settings.update_type == 1:
            self.grid.randomize()
            self.plot_updated = True
        if Settings.collision_avoidance == True:

            self.collision_avoidance.update_obstacles(self.grid.get_obstacles()[1], self.grid.range_scale)
            self.wp_straight_ahead_clicked()
        else:
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

    @QtCore.pyqtSlot()
    def run(self):
        try:
            trans = self.grid.trans(self.diff.dx, self.diff.dy)
            rot = self.grid.rot(self.diff.dpsi)
            self.grid.calc_obstacles()
            # self.signals.finished.emit(trans or rot)
            self.signals.finished.emit(True)
            # print('Translate: {}\tRotate: {}'.format(trans, rot))
        except Exception as e:
            logger.error('Grid Worker', e)

    def update(self, diff):
        self.diff = diff

class GridWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(bool, name='grid_worker_finished')


if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

    if Settings.collision_avoidance:
        window.login_widget.collision_avoidance.save_paths()
    if Settings.show_voronoi_plot:
        window.login_widget.voronoi_window.close()
    if Settings.save_scan_lines:
        np.savez('pySonarLog/scan_lines_{}'.format(strftime("%Y%m%d-%H%M%S")), scan_lines=np.array(window.login_widget.scan_lines))
    if Settings.input_source == 0:
        window.login_widget.udp_client.close()
