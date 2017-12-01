import numpy as np
import math
import logging


logger = logging.getLogger('RawPlot')

class RawPlot(object):
    PI2 = math.pi/2
    def __init__(self, l_x, l_y, resolution):
        """

        :param l_x: length in x direction
        :param l_y: length in y direction
        :param resolution: resolution in pixels pr meter
        """
        self.x_max = math.floor(resolution*l_x)
        self.y_max = math.floor(resolution*l_y)
        if self.x_max % 2 == 0:
            self.x_max += 1
            logger.info('Extended grid by one cell in X direction to make it even')
        half_cell_size = 1 / (resolution*2)
        origo_j = math.floor(self.x_max/2)
        self.grid = np.zeros((self.y_max, self.x_max), dtype=np.uint8)
        self.cell_angle = np.zeros((self.y_max, self.x_max))
        self.cell_rad = np.zeros((self.y_max, self.x_max))
        self.thetaLow = np.zeros((self.y_max, self.x_max))
        self.thetaHigh = np.zeros((self.y_max, self.x_max))
        self.rLow = np.zeros((self.y_max, self.x_max))
        self.rHigh = np.zeros((self.y_max, self.x_max))
        cell_x_value, cell_y_value = np.meshgrid(np.linspace(-l_x/2, l_x/2, l_x*resolution+1),
                                                           np.linspace(l_y, 0, l_y*resolution), indexing='xy')
        self.cell_rad = np.sqrt(cell_x_value ** 2 + cell_y_value ** 2)
        self.cell_angle = np.arctan2(cell_x_value, cell_y_value)

        # ranges
        self.rHigh = np.sqrt((cell_x_value +
                              np.sign(cell_x_value) * half_cell_size) ** 2
                             + (cell_y_value + half_cell_size) ** 2)
        self.rLow = np.sqrt((cell_x_value - np.sign(cell_x_value) * half_cell_size) ** 2 + (
            np.fmax(cell_y_value - half_cell_size, 0)) ** 2)

        # angles x<0
        self.thetaLow[:, :origo_j] = np.arctan2(cell_x_value[:, :origo_j] - half_cell_size,
                                                    cell_y_value[:, :origo_j] - half_cell_size)
        self.thetaHigh[:, :origo_j] = np.arctan2(cell_x_value[:, :origo_j] + half_cell_size,
                                                     cell_y_value[:, :origo_j] + half_cell_size)
        self.thetaLow[:, origo_j + 1:] = np.arctan2(cell_x_value[:, origo_j + 1:] - half_cell_size,
                                                        cell_y_value[:, origo_j + 1:] + half_cell_size)
        self.thetaHigh[:, origo_j + 1:] = np.arctan2(cell_x_value[:, origo_j + 1:] + half_cell_size,
                                                         cell_y_value[:, origo_j + 1:] - half_cell_size)

        self.thetaLow[:, origo_j] = np.arctan2(cell_x_value[:, origo_j] - half_cell_size,
                                                   cell_y_value[:, origo_j] - half_cell_size)
        self.thetaHigh[:, origo_j] = np.arctan2(cell_x_value[:, origo_j] + half_cell_size,
                                                    cell_y_value[:, origo_j] - half_cell_size)

    def update_grid(self, msg):
        theta1 = max(msg.bearing - msg.step / 2, -self.PI2)
        theta2 = min(msg.bearing + msg.step / 2, self.PI2)
        cone = np.asarray(np.nonzero((self.thetaHigh.flat >= theta1) & (self.thetaLow.flat <= theta2)))
        dx = msg.rangeScale/msg.dataBins
        self.grid.flat[cone] = 1
        for i in range(msg.dataBins):
            bin_i = cone[np.nonzero((self.rLow.flat[cone] >= (i - 0.5) * dx) & (self.rHigh.flat[cone] <= (i + 0.5) * dx))]
            self.grid.flat[bin_i] = msg.data[i]

    def clear(self):
        self.grid = np.zeros(np.shape(self.grid))
