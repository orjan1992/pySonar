class Settings:
    # 0 == udp, 1 == MOOS
    input_source = 1
    # 0 == raw_plot, 1 == prob_plot, 2 == obstacle_plot
    plot_type = 2
    # 0 == raw update, 1 == zhou update
    update_type = 0


class GridSettings:
        half_grid = True
        p_inital = 0.65
        binary_threshold = 0.78
        binary_grid = False


class ConnectionSettings:
        sonar_port = 5555


class PlotSettings:
        steps = [0, 0.33, 0.67, 1]
        colors = [[0.2, 0.2, 0.2, 0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        max_val = 50.0
        min_val = -50.0
        threshold = 0

