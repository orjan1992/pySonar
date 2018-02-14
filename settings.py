import cv2


class Settings:
    # 0 == udp, 1 == MOOS
    input_source = 1
    # 0 == raw_plot, 1 == prob_plot, 2 == obstacle_plot
    plot_type = 0
    # 0 == raw update, 1 == zhou update
    update_type = 0
    pos_update = 1000.0/60.0
    hist_window = False


class GridSettings:
        half_grid = True
        p_inital = 0
        binary_threshold = 0.78
        binary_grid = False


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