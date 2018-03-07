import cv2
import numpy as np
from PyQt5.QtGui import QColor, QBrush, QPen

class Settings:
    # 0 == udp, 1 == MOOS
    input_source = 1
    # 0 == raw_plot, 1 == prob_plot, 2 == obstacle_plot
    plot_type = 2
    # 0 == raw update, 1 == zhou update
    update_type = 0
    pos_update = 1000.0/60.0  # ms
    hist_window = False
    collision_avoidance = True
    show_map = True
    show_voronoi_plot = False
    collision_avoidance_interval = 1000  # ms
    save_obstacles = False


class CollisionSettings:
    border_step = 30
    wp_as_gen_point = False
    obstacle_margin = 3 # meter
    loop_sleep = 0.001
    max_loop_iterations = 100
    parallel_line_tolerance = 2*np.pi/180.0
    send_new_wps = True
    start_penalty_factor = 1000.0

    fermat_kappa_max = 1  # max curvature
    fermat_step_factor = 0.8


class MapSettings:
    display_grid = True
    grid_dist = 10
    grid_pen = QPen(QColor(198, 198, 236))
    grid_center_pen = QPen(QColor(255, 0, 0))

    sonar_obstacle_pen = QPen(QColor(0, 0, 255))

    waypoint_size = 10.0
    waypoint_active_color = QColor(0, 255, 0, 255)
    waypoint_active_pen = QPen(waypoint_active_color)
    waypoint_inactive_color = QColor(255, 102, 0, 255)
    waypoint_inactive_pen = QPen(waypoint_inactive_color)
    avoidance_waypoint_pen = QPen(QColor(255, 0, 0))

    vehicle_size = 10.0
    vehicle_form_factor = 0.4
    vehicle_color = QColor(255, 0, 0, 255)
    vehicle_pen = QPen(vehicle_color)
    vehicle_pen.setWidth(2)
    vehicle_brush = QBrush(vehicle_color)

    sonar_circle_brush = QBrush(QColor(68, 198, 250, 50))

    obstacle_color = QColor(0, 0, 0, 255)
    obstacle_pen = QPen(obstacle_color)
    obstacle_pen.setWidth(2)
    obstacle_brush = QBrush(obstacle_color)

class GridSettings:
        half_grid = True
        p_inital = 0
        binary_threshold = 0.78
        binary_grid = False
        width = 1601
        height = 801
        min_set_pixels = 1601.0*801.0/3.0

class FeatureExtraction:
    kernel = np.ones((11, 11), dtype=np.uint8)
    iterations = 1
    min_area = 20

class ConnectionSettings:
        sonar_port = 5555


class PlotSettings:
        steps = [0, 0.33, 0.67, 1]
        colors = [[0.2, 0.2, 0.2, 0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        max_val = 50.0
        min_val = -50.0
        threshold = 10
