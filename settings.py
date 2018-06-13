import cv2
import numpy as np
from PyQt5.QtGui import QColor, QBrush, QPen
from pyproj import Proj

class Settings:
    save_obstacles = False
    save_paths = False
    save_collision_info = True

    enable_autopilot = False
    auto_settings = True
    # 0 == udp, 1 == MOOS
    input_source = 1
    pos_msg_source = 1 # 0=NMEA, 1=Autopilot
    # 0 == raw_plot, 1 == prob_plot, 2 == obstacle_plot
    plot_type = 2
    # 0 == raw update, 1 == zhou update
    update_type = 1
    pos_update_speed = 100  # ms

    if input_source == 0:
        log_folder = 'C:/Users/Ørjan/Desktop/logs/'
    else:
        log_folder = '/home/orjangr/Repos/pySonar/pySonarLog/sim_log/'

    hist_window = False
    collision_avoidance = True
    if input_source == 0:
        show_map = False
    else:
        show_map = True
    show_wp_on_grid = True
    show_voronoi_plot = False
    show_pos = True
    collision_avoidance_interval = 200  # ms
    save_scan_lines = False


    button_height = 30
    button_width = 200
    inverted_sonar = False

    ## Auto settings
    if auto_settings:
        if enable_autopilot:
            pos_msg_source = 1
            # collision_avoidance = True
        else:
            pos_msg_source = 0
            # collision_avoidance = False


class CollisionSettings:
    border_step = 100
    wp_as_gen_point = False
    obstacle_margin = 2 # meter
    vehicle_margin = 2 # meter
    send_new_wps = True

    fermat_kappa_max = 0.5  # max curvature
    fermat_step_factor = 0.8

    first_wp_dist = 4 # meters

    colinear_angle = 2*np.pi/180.0

    dummy_wp_factor = (2, 0)
    use_fermat = True
    cubic_smoothing_discrete_step = 0.1

    if Settings.auto_settings:
        if Settings.enable_autopilot:
            use_fermat = True
            # send_new_wps = True
        else:
            use_fermat = True
            # send_new_wps = False

class LosSettings:
    enable_los = True
    cruise_speed = 0.4
    look_ahead_time = 20
    roa = 2
    # look_ahead_distance =
    send_new_heading_limit = 0.1*np.pi/180.0  # 0.5*np.pi/180.0

    max_heading_change = 5*np.pi/180.0  # Rad/m
    max_acc = 0.12
    safe_turning_speed = 0.25
    braking_distance = 5

    start_heading_diff = 5*np.pi/180.0
    log_paths = True

    if Settings.auto_settings:
        if Settings.enable_autopilot:
            enable_los = True
        else:
            enable_los = False

    # inital_wp_list = [[6821801, 458050, 2, 0.5],
    #                 [6821790, 458030, 2, 0.5],
    #                 [6821760, 458060, 2, 0.5],
    #                 [6821740, 458080, 2, 0.5],
    #                 [6821740, 458110, 2, 0.5],
    #                 [6821750, 458125, 2, 0.5],
    #                 [6821730, 458135, 2, 0.5],
    #                 [6821707, 458065, 2, 0.5],
    #                 [6821680, 458035, 2, 0.5],
    #                 [6821676, 458080, 2, 0.5],
    #                 [6821750, 458100, 2, 0.5],
    #                 [6821795, 458060, 2, 0.5]]
    # inital_wp_list = np.load('collision_avoidance/smooth.npz')['smooth']

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
    if Settings.input_source == 0:
        threshold = 170
    else:
        threshold = 40
    half_grid = False
    p_inital = 0.5
    p_binary_threshold = 0.6
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
    min_rot = 0.5*np.pi/180  # in rad
    min_trans = 1  # in pixels
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
    randomize_max = 0.3

class FeatureExtraction:
    kernel = np.ones((11, 11), dtype=np.uint8)
    iterations = 1
    # min_area = 9  # pixels/m
    min_area = 4  # pixels/m

class ConnectionSettings:
        sonar_port = 4002
        pos_port = 4006
        # sonar_port = 4001
        # pos_port = 4005


        # Other settings
        autopilot_server_port = 4010
        autopilot_listen_port = 4015
        autopilot_ip = '10.3.2.40'
        # autopilot_ip = '0.0.0.0'
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
        vehicle_width_drawing_factor = 0.5

class Map:
    map_proj = Proj(proj='utm', zone='31N', ellps='WGS84')
    # map_proj = Proj(proj='utm', zone='31N', ellps='intl')
    apply_pos_offset = True
    # pos_offset = [-19.535787283442914, -86.253202902036720]
    pos_offset = [0, 0]
    # map_proj = Proj(proj='utm', zone=31, ellps='WGS84')

