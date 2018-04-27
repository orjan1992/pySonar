import cv2
import numpy as np
from PyQt5.QtGui import QColor, QBrush, QPen
from pyproj import Proj

class Settings:
    # 0 == udp, 1 == MOOS
    input_source = 0
    pos_msg_source = 0 # 0=NMEA, 1=Autopilot
    # 0 == raw_plot, 1 == prob_plot, 2 == obstacle_plot
    plot_type = 2
    # 0 == raw update, 1 == zhou update
    update_type = 1
    pos_update_speed = 100  # ms

    enable_autopilot = False

    hist_window = False
    collision_avoidance = True
    show_map = False
    show_wp_on_grid = True
    show_voronoi_plot = False
    show_pos = True
    collision_avoidance_interval = 200  # ms

    save_obstacles = False
    save_paths = True
    save_scan_lines = False
    save_collision_info = True

    button_height = 30
    button_width = 200
    inverted_sonar = False


class CollisionSettings:
    border_step = 100
    wp_as_gen_point = False
    obstacle_margin = 2 # meter
    vehicle_margin = 1 # meter
    send_new_wps = False

    fermat_kappa_max = 0.5  # max curvature
    fermat_step_factor = 0.8

    first_wp_dist = 4 # meters

    tracking_speed = 0.5
    dummy_wp_factor = (2, 0)
    use_fermat = True
    cubic_smoothing_discrete_step = 0.1

class LosSettings:
    enable_los = False
    cruise_speed = 0.5
    look_ahead_time = 8
    roa = 2
    # look_ahead_distance =
    send_new_heading_limit = 0.5*np.pi/180.0

    max_heading_change = 5*np.pi/180.0  # Rad/m
    max_acc = 0.12
    safe_turning_speed = 0.25
    braking_distance = 5

    start_heading_diff = 1*np.pi/180.0


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
    threshold = 200
    half_grid = False
    p_inital = 0.5
    p_binary_threshold = 0.7
    p_free = 0.3
    p_occ = 0.71
    hit_factor = 50

    binary_grid = False
    width = 1601
    if half_grid:
        height = 801
    else:
        height = 1601
    max_unset_pixels = 1601.0 * 801.0 / 4.0
    min_rot = 4  # in 1/16 grad
    cell_factor = 16
    scale_raw_data = False
    smoothing_factor = 10  # 10 for real data

    # zhou model
    kh_high = 0.5445427266222308
    # P_DI_min_high = np.sin(0.5 * kh_high * np.sin(1.5 * np.pi / 180.0)) / (0.5 * kh_high * np.sin(1.5 * np.pi / 180.0))
    # P_DI_max_high = np.sin(0.5 * kh_high * np.sin(0.000000001)) / (0.5 * kh_high * np.sin(0.000000001))

    kh_low = 0.2722713633111154
    # P_DI_min_low = np.sin(0.5 * kh_low * np.sin(3 * np.pi / 180.0)) / (0.5 * kh_low * np.sin(3 * np.pi / 180.0))
    # P_DI_max_low = np.sin(0.5 * kh_low * np.sin(0.000000001)) / (0.5 * kh_low * np.sin(0.000000001))

    mu = 1
    randomize_size = 8
    randomize_max = 0.7

class FeatureExtraction:
    kernel = np.ones((11, 11), dtype=np.uint8)
    iterations = 1
    min_area = 9  # pixels/m

class ConnectionSettings:
        sonar_port = 4002
        pos_port = 4006
        # sonar_port = 4001
        # pos_port = 4005


        # Other settings
        autopilot_server_port = 4015
        autopilot_listen_port = 4010
        autopilot_ip = '127.0.0.1'
        use_nmea_checksum = False
        autopilot_watchdog_timeout = 0.5


class PlotSettings:
        steps_raw = [0, 0.33, 0.67, 1]
        steps_prob = [-5, -1.67, 1.67, 5]
        colors = [[0.2, 0.2, 0.2, 0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        max_val = 50.0
        min_val = -50.0

        wp_on_grid_color = (0, 153, 0)
        wp_on_grid_thickness = 4
        wp_on_grid_radius = 20

class Map:
    map_proj = Proj(proj='utm', zone=31, ellps='WGS84')