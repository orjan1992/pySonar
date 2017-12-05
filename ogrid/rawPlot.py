import numpy as np
import math
from pathlib import Path
import logging
import os

logger = logging.getLogger('OGrid')


class RawPlot(object):
    cur_step = 0
    GRAD2RAD = math.pi / (16 * 200)
    RAD2GRAD = (16 * 200) / math.pi
    PI2 = math.pi / 2
    PI4 = math.pi / 4
    DEG1_5 = 1.5*math.pi/180
    DEG7_5 = DEG1_5/2
    oLog_type = np.float32
    old_delta_x = 0
    old_delta_y = 0
    old_delta_psi = 0
    MIN_ROT = 0.1 * math.pi / 180
    mapping_data_type = np.uint32

    # TODO fix quickfix with fixed ranges and bins
    upper_range = np.linspace(0, 30, 300) + 0.05
    lower_range = np.copy(upper_range) - 0.1

    def __init__(self, cellSize, sizeX, sizeY):
        if cellSize > 0:
            if (sizeX > cellSize) or (sizeY > cellSize):
                if round(sizeX / cellSize) % 2 == 0:
                    sizeX = sizeX + cellSize
                    logger.info('Extended grid by one cell in X direction to make it even')
                self.XLimMeters = sizeX / 2
                self.YLimMeters = sizeY
                self.cellSize = cellSize
                self.fourth_cell_size = cellSize / 4
                self.half_cell_size = cellSize / 2
                self.cellArea = cellSize ** 2
                self.X = round(sizeX / cellSize)
                self.Y = round(sizeY / cellSize)
                self.origoJ = round(self.X / 2)
                self.origoI = self.Y
                self.grid = np.zeros((self.Y, self.X), dtype=np.double)
                [self.iMax, self.jMax] = np.shape(self.grid)

                fStr = 'OGrid_data/angleRad_X=%i_Y=%i_size=%i.npz' % (self.X, self.Y, int(cellSize * 100))
                try:
                    tmp = np.load(fStr)
                    self.r = tmp['r']
                    self.rHigh = tmp['rHigh']
                    self.rLow = tmp['rLow']
                    self.theta = tmp['theta']
                    self.thetaHigh = tmp['thetaHigh']
                    self.thetaLow = tmp['thetaLow']
                    self.cell_x_value = tmp['cell_x_value']
                    self.cell_y_value = tmp['cell_y_value']
                except FileNotFoundError:
                    # Calculate angles and radii
                    self.r = np.zeros((self.Y, self.X))
                    self.rHigh = np.zeros((self.Y, self.X))
                    self.rLow = np.zeros((self.Y, self.X))
                    self.theta = np.zeros((self.Y, self.X))
                    self.thetaHigh = np.zeros((self.Y, self.X))
                    self.thetaLow = np.zeros((self.Y, self.X))
                    self.cell_x_value = np.zeros((self.Y, self.X))
                    self.cell_y_value = np.zeros((self.Y, self.X))
                    # x = np.arrange((-self.origoJ * self.cellSize), (self.origoJ * self.cellSize), self.cellSize)
                    # y =
                    for i in range(0, self.Y):
                        for j in range(0, self.X):
                            self.cell_x_value[i, j] = (j - self.origoJ) * self.cellSize
                            self.cell_y_value[i, j] = (self.origoI - i) * self.cellSize

                    self.r = np.sqrt(self.cell_x_value ** 2 + self.cell_y_value ** 2)
                    self.theta = np.arctan2(self.cell_x_value, self.cell_y_value)
                    # ranges
                    self.rHigh = np.sqrt((self.cell_x_value +
                                          np.sign(self.cell_x_value) * self.half_cell_size) ** 2
                                         + (self.cell_y_value + self.half_cell_size) ** 2)
                    self.rLow = np.sqrt((self.cell_x_value - np.sign(self.cell_x_value) * self.half_cell_size) ** 2 + (
                    np.fmax(self.cell_y_value - self.half_cell_size, 0)) ** 2)

                    # angles x<0
                    self.thetaLow[:, :self.origoJ] = np.arctan2(
                        self.cell_x_value[:, :self.origoJ] - self.half_cell_size,
                        self.cell_y_value[:, :self.origoJ] - self.half_cell_size)
                    self.thetaHigh[:, :self.origoJ] = np.arctan2(
                        self.cell_x_value[:, :self.origoJ] + self.half_cell_size,
                        self.cell_y_value[:, :self.origoJ] + self.half_cell_size)
                    self.thetaLow[:, self.origoJ + 1:] = np.arctan2(
                        self.cell_x_value[:, self.origoJ + 1:] - self.half_cell_size,
                        self.cell_y_value[:, self.origoJ + 1:] + self.half_cell_size)
                    self.thetaHigh[:, self.origoJ + 1:] = np.arctan2(
                        self.cell_x_value[:, self.origoJ + 1:] + self.half_cell_size,
                        self.cell_y_value[:, self.origoJ + 1:] - self.half_cell_size)

                    self.thetaLow[:, self.origoJ] = np.arctan2(self.cell_x_value[:, self.origoJ] - self.half_cell_size,
                                                               self.cell_y_value[:, self.origoJ] - self.half_cell_size)
                    self.thetaHigh[:, self.origoJ] = np.arctan2(self.cell_x_value[:, self.origoJ] + self.half_cell_size,
                                                                self.cell_y_value[:, self.origoJ] - self.half_cell_size)
                    if not os.path.isdir('OGrid_data'):
                        logger.info('Made a new directory for data files.')
                        os.makedirs('OGrid_data')
                    np.savez(fStr, r=self.r, rHigh=self.rHigh, rLow=self.rLow, theta=self.theta,
                             thetaHigh=self.thetaHigh, thetaLow=self.thetaLow, cell_x_value=self.cell_x_value,
                             cell_y_value=self.cell_y_value)
            self.MAX_ROT = np.min(
                np.abs(np.arcsin((self.cellSize + self.cell_x_value[0, 0]) / self.r[0, 0]) - self.theta[0, 0]),
                np.abs(np.arccos((self.cellSize + self.cell_y_value[0, -1]) / self.r[0, -1]) - self.theta[0, -1]))
            self.MAX_ROT_BEFORE_RESET = 30 * self.MAX_ROT
            # self.steps = np.array([4, 8, 16, 32])
            self.steps = np.array([16]) # TODO change steps back again
            self.bearing_ref = np.linspace(-self.PI2, self.PI2, self.RAD2GRAD * math.pi)
            self.mappingMax = int(self.X * self.Y / 10)
            self.makeMap(self.steps)
            self.loadMap(self.steps[0] * self.GRAD2RAD)
            self.deltaSurface = 0  # 1.5*self.cellSize
            self.cellSize_with_margin = self.cellSize * 1.01

    def makeMap(self, step_angle_size):
        filename_base = 'OGrid_data/Step_X=%i_Y=%i_size=%i_step=' % (self.X, self.Y, int(self.cellSize*100))
        steps_to_create = []
        for i in range(0, step_angle_size.shape[0]):
            if not Path('%s%i.npz' % (filename_base, step_angle_size[i])).is_file():
                steps_to_create.append(step_angle_size[i])
        if steps_to_create:
            logger.info('Need to create %i steps' % len(steps_to_create))
            k = 1
            # Create  Mapping
            step = np.array(steps_to_create) * self.GRAD2RAD
            for j in range(0, len(step)):
                mapping = np.zeros((len(self.bearing_ref), self.mappingMax), dtype=self.mapping_data_type)
                for i in range(0, len(self.bearing_ref)):
                    cells = self.sonarCone(step[j], self.bearing_ref[i])
                    try:
                        mapping[i, 0:len(cells)] = cells
                    except ValueError as error:
                        raise MyException('Mapping variable to small !!!!')
                if np.max(np.max(mapping)) == np.iinfo(self.mapping_data_type).max:
                    raise MyException('Mapping data type is to small')
                # Saving to file
                np.savez('%s%i.npz' % (filename_base, steps_to_create[j]), mapping=mapping)
                logger.info('Step %i done!' % k)
                k += 1

    def loadMap(self, step):
        # LOADMAP Loads the map. Step is in rad
        step = round(step * self.RAD2GRAD)
        if self.cur_step != step or not self.mapping.any():
            if not any(np.nonzero(self.steps == step)):
                self.makeMap(np.array([step]))
            try:
                self.mapping = np.load(
                    'OGrid_data/Step_X=%i_Y=%i_size=%i_step=%i.npz' % (self.X, self.Y, int(self.cellSize * 100), step))[
                    'mapping']
            except FileNotFoundError:
                logger.error('Could not find mapping file!: {s}'.format('OGrid_data/Step_X=%i_Y=%i_size=%i_step=%i.npz'
                                                                        % (self.X, self.Y, int(self.cellSize * 100),
                                                                           step)))
                raise MyException('Could not find mapping file!')
            self.cur_step = step

    def sonarCone(self, step, theta):
        # TODO should be optimized
        # TODO option for 650kHz(1.5 deg x 40 deg beam) and 325kHz(3 deg x 20 deg beam)
        theta1 = max(theta - self.DEG7_5, -self.PI2)
        theta2 = min(theta + self.DEG7_5, self.PI2)
        a = np.ravel_multi_index(np.nonzero(self.thetaLow <= theta2), (self.iMax, self.jMax))
        b = np.ravel_multi_index(np.nonzero(self.thetaHigh >= theta1), (self.iMax, self.jMax))
        return np.intersect1d(a, b)

    def sonarConeLookup(self, step, theta):
        # step is in rad
        self.loadMap(step)
        if np.min(np.absolute(theta - self.bearing_ref)) > step * 0.5:
            logger.error('Difference between theta and theta ref in sonarConeLookup is {}'.format(
                np.min(np.absolute(theta - self.bearing_ref))))
            raise MyException('Difference between theta and theta ref is to large!')
        j = np.argmin(np.absolute(theta - self.bearing_ref))
        cone = self.mapping[j]
        return cone[cone != 0]

    def autoUpdate(self, msg):
        dl = msg.rangeScale / np.shape(msg.data)[0]
        theta = msg.bearing
        cone = self.sonarConeLookup(msg.step, theta)
        # self.grid.flat[cone] = 1
        # if msg.dataBins == 300 and msg.rangeScale == 30:
        #     for j in range(0, 300):
        #         self.grid.flat[cone[np.nonzero(self.rLow.flat[cone] < self.upper_range[j] and self.rHigh > self.lower_range[j]]]\
        #             = msg.data[j]
        # else:

        for j in range(1, len(msg.data)):
            range_scale = j*dl
            if abs((range_scale) * math.sin(theta)) > self.XLimMeters or abs((range_scale) * math.cos(theta)) > self.YLimMeters:
                break  # SJEKK DETTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.grid.flat[(cone[self.rLow.flat[cone] < range_scale + dl/2])] = msg.data[j]
            cone = cone[self.rHigh.flat[cone] >= (range_scale - dl/2)]

    def clearGrid(self):
        self.grid = np.zeros(np.shape(self.grid))
        logger.info('Grid cleared')

# Exeption class for makin understanable exception
class MyException(Exception):
    pass
