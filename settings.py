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
    pos_update = 1000.0/60.0
    hist_window = False
    collision_avoidance = True
    show_map = True
    collision_avoidance_interval = 1000.0/2.0


class CollisionSettings:
    border_step = 30


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


class BlobDetectorSettings:
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 50
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = False
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
