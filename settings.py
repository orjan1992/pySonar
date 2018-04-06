import cv2
import numpy as np
from PyQt5.QtGui import QColor, QBrush, QPen

class Settings:
    # 0 == udp, 1 == MOOS
    input_source = 0
    # 0 == raw_plot, 1 == prob_plot, 2 == obstacle_plot
    plot_type = 0
    # 0 == raw update, 1 == zhou update
    update_type = 0
    pos_update = 1000.0/60.0  # ms
    hist_window = False
    collision_avoidance = False
    show_map = False
    show_voronoi_plot = False
    collision_avoidance_interval = 200  # ms
    save_obstacles = False
    save_paths = False
    save_scan_lines = False


class CollisionSettings:
    border_step = 30
    wp_as_gen_point = False
    obstacle_margin = 2 # meter
    vehicle_margin = 1 # meter
    loop_sleep = 0.001
    max_loop_iterations = 100
    parallel_line_tolerance = 2*np.pi/180.0
    send_new_wps = True
    add_path_deviation_penalty = False
    start_penalty_factor = 1000.0
    path_deviation_penalty_factor = 10.0

    fermat_kappa_max = 0.5  # max curvature
    fermat_step_factor = 0.8

    first_wp_dist = 1


class MapSettings:
    display_grid = True
    show_collision_margins = True
    grid_dist = 10
    grid_pen = QPen(QColor(198, 198, 236))
    grid_center_pen = QPen(QColor(255, 0, 0))

    sonar_obstacle_pen = QPen(QColor(0, 0, 255))
    sonar_collision_margin_pen = QPen(QColor(255, 0, 0))

    waypoint_size = 10.0
    waypoint_active_color = QColor(0, 255, 0, 255)
    waypoint_active_pen = QPen(waypoint_active_color)
    waypoint_inactive_color = QColor(255, 102, 0, 255)
    waypoint_inactive_pen = QPen(waypoint_inactive_color)
    avoidance_waypoint_pen = QPen(QColor(255, 0, 0))
    waypoint_invalid_pen = QPen(QColor(255, 0, 0))

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
    half_grid = False
    p_inital = 0.5
    p_binary_threshold = 0.78
    p_free = 0.3
    p_occ = 0.9
    hit_factor = 5

    binary_grid = False
    width = 1601
    if half_grid:
        height = 801
    else:
        height = 1601
    min_set_pixels = 1601.0*801.0/3.0
    cell_factor = 16
    scale_raw_data = True

class FeatureExtraction:
    kernel = np.ones((11, 11), dtype=np.uint8)
    iterations = 1
    min_area = 30

class ConnectionSettings:
        sonar_port = 4001
        # pos_port = 4005
        pos_port = 40010
        wp_port = 5000
        wp_ip = None
        use_nmea_checksum = False


class PlotSettings:
        steps_raw = [0, 0.33, 0.67, 1]
        steps_prob = [-5, -1.67, 1.67, 5]
        colors = [[0.2, 0.2, 0.2, 0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        max_val = 50.0
        min_val = -50.0
        threshold = 10
