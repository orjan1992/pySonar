class Settings:
    input_source = 1
    raw_plot = True


class GridSettings:
        half_grid = 1
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
        threshold = 40

