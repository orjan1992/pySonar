import numpy as np
import math
from pathlib import Path
import logging
import os
import inspect


logger = logging.getLogger('OGrid')

class OGrid(object):
    cur_step = 0
    GRAD2RAD = math.pi / (16 * 200)
    RAD2GRAD = (16 * 200) / math.pi
    PI2 = math.pi / 2
    PI4 = math.pi / 4
    oLog_type = np.float32
    old_delta_x = 0
    old_delta_y = 0
    old_delta_psi = 0
    MIN_ROT = 0.1*math.pi/180

    def __init__(self, cellSize, sizeX, sizeY, p_m, binary_threshold=0.7):
        if cellSize > 0:
            if (sizeX > cellSize) or (sizeY > cellSize):
                if round(sizeX / cellSize) % 2 == 0:
                    sizeX = sizeX + cellSize
                    logger.info('Extended grid by one cell in X direction to make it even')
                self.XLimMeters = sizeX / 2
                self.YLimMeters = sizeY
                self.cellSize = cellSize
                self.fourth_cell_size = cellSize/4
                self.half_cell_size = cellSize/2
                self.cellArea = cellSize ** 2
                self.X = round(sizeX / cellSize)
                self.Y = round(sizeY / cellSize)
                self.origoJ = round(self.X / 2)
                self.origoI = self.Y
                self.OZero = math.log(p_m / (1 - p_m))
                self.binary_threshold = math.log(binary_threshold/(1-binary_threshold))
                self.oLog = np.ones((self.Y, self.X), dtype=self.oLog_type) * self.OZero
                self.O_logic = np.zeros((self.Y, self.X), dtype=bool)
                [self.iMax, self.jMax] = np.shape(self.oLog)


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
                    self.rLow = np.sqrt((self.cell_x_value - np.sign(self.cell_x_value) * self.half_cell_size) ** 2 + (np.fmax(self.cell_y_value - self.half_cell_size, 0)) ** 2)

                    # angles x<0
                    self.thetaLow[:, :self.origoJ] = np.arctan2(self.cell_x_value[:, :self.origoJ] - self.half_cell_size, 
                                                              self.cell_y_value[:, :self.origoJ] - self.half_cell_size)
                    self.thetaHigh[:, :self.origoJ] = np.arctan2(self.cell_x_value[:, :self.origoJ] + self.half_cell_size,
                                                               self.cell_y_value[:, :self.origoJ] + self.half_cell_size)
                    self.thetaLow[:, self.origoJ+1:] = np.arctan2(self.cell_x_value[:, self.origoJ+1:] - self.half_cell_size,
                                                                self.cell_y_value[:, self.origoJ+1:] + self.half_cell_size)
                    self.thetaHigh[:, self.origoJ+1:] = np.arctan2(self.cell_x_value[:, self.origoJ+1:] + self.half_cell_size,
                                                                 self.cell_y_value[:, self.origoJ+1:] - self.half_cell_size)

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
            self.MAX_ROT = np.min(np.abs(np.arcsin((self.cellSize+self.cell_x_value[0, 0])/self.r[0, 0])-self.theta[0, 0]),
                                  np.abs(np.arccos((self.cellSize+self.cell_y_value[0, -1])/self.r[0, -1])-self.theta[0, -1]))
            self.MAX_ROT_BEFORE_RESET = 30*self.MAX_ROT
            self.steps = np.array([4, 8, 16, 32])
            self.bearing_ref = np.linspace(-self.PI2, self.PI2, self.RAD2GRAD * math.pi)
            self.mappingMax = int(self.X * self.Y / 10)
            self.makeMap(self.steps)
            self.loadMap(self.steps[0] * self.GRAD2RAD)
            self.deltaSurface = 1.5*self.cellSize
            self.cellSize_with_margin = self.cellSize*1.01

    def makeMap(self, step_angle_size):
        filename_base = 'OGrid_data/Step_X=%i_Y=%i_size=%i_step=' % (self.X, self.Y, int(self.cellSize * 100))
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
                mapping = np.zeros((len(self.bearing_ref), self.mappingMax), dtype=np.uint16)
                for i in range(0, len(self.bearing_ref)):
                    cells = self.sonarCone(step[j], self.bearing_ref[i])
                    try:
                        mapping[i, 0:len(cells)] = cells
                    except ValueError as error:
                        raise MyException('Mapping variable to small !!!!')
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
                                                                        % (self.X, self.Y, int(self.cellSize * 100), step)))
                raise MyException('Could not find mapping file!')
            self.cur_step = step

    def sonarCone(self, step, theta):
        # TODO should be optimized
        theta1 = max(theta - step / 2, -self.PI2)
        theta2 = min(theta + step / 2, self.PI2)
        (row, col) = np.nonzero(self.thetaLow <= theta2)
        a = self.sub2ind(row, col)

        (row, col) = np.nonzero(self.thetaHigh >= theta1)
        b = self.sub2ind(row, col)
        return np.intersect1d(a, b)

    def sub2ind(self, row, col):
        return col + row * self.jMax

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

    def updateCells(self, cells, value):
        for cell in cells:
            self.oLog.flat[cell] = value

    def getP(self):
        try:
            P = 1 - 1 / (1 + np.exp(self.oLog))
        except RuntimeWarning:
            logger.debug('Overflow when calculating probability')
        return P

    def get_binary_map(self):
        return (self.oLog > self.binary_threshold).astype(np.float)

    def updateCellsZhou2(self, cone, rangeScale, theta):
        # UPDATECELLSZHOU
        subRange = cone[self.rHigh.flat[cone] < rangeScale - self.deltaSurface]
        onRange = cone[self.rLow.flat[cone] < (rangeScale + self.deltaSurface)]
        onRange = onRange[self.rHigh.flat[onRange] > (rangeScale - self.deltaSurface)]

        self.oLog.flat[subRange] -= 2.19722  #self.OZero  # 4.595119850134590

        alpha = np.abs(theta - self.theta.flat[onRange])
        kh2 = 0.5  # MÅ defineres
        mu = 1  # MÅ Defineres
        P_DI = np.sin(kh2 * np.sin(alpha)) / (kh2 * np.sin(alpha)+0.00000000001)
        P_TS = 0.7
        minP = 0
        maxP = 1
        P = P_DI * P_TS
        P_O = (P - minP) / (2 * (maxP - minP)) + 0.5
        self.oLog.flat[onRange] += np.log(P_O / (1 - P_O)) + self.OZero
        return cone[self.rLow.flat[cone] >= (rangeScale + self.deltaSurface)]

    def autoUpdateZhou(self, msg, threshold):
        dl = msg.rangeScale / np.shape(msg.data)[0]
        theta = msg.bearing
        not_updated_cells = self.sonarConeLookup(msg.step, theta)
        distance_updated = False
        for j in range(1, len(msg.data)):
            if abs((j * dl) * math.sin(theta)) > self.XLimMeters or abs((j * dl) * math.cos(theta)) > self.YLimMeters:
                break  # SJEKK DETTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if msg.data[j] > threshold:
                not_updated_cells = self.updateCellsZhou2(not_updated_cells, j * dl, theta)
                distance_updated = True
        if not distance_updated:
            self.updateCellsZhou2(not_updated_cells, math.inf, theta)

    def clearGrid(self):
        self.oLog = np.ones((self.Y, self.X)) * self.OZero
        self.O_logic = np.zeros((self.Y, self.X), dtype=bool)
        logger.info('Grid cleared')

    def translational_motion(self, delta_x, delta_y):
        """
        transform grid for deterministic translational motion
        :param delta_x: Change in positive grid direction [m] (sway)
        :param delta_y: Change in positive grid direction [m] (surge)
        :return: Nothing
        """
        # Check if movement is less than 1/4 of cell size => save for later
        delta_x += self.old_delta_x
        delta_y += self.old_delta_y
        if abs(delta_x) < self.fourth_cell_size:
            self.old_delta_x = delta_x
            delta_x = 0
        if abs(delta_y) < self.fourth_cell_size:
            self.old_delta_y = delta_y
            delta_y = 0
        if delta_y == delta_x == 0:
            return

        # Check if movement is > cell size => new itteration
        new_iteration_needed = False
        new_delta_y = 0
        new_delta_x = 0
        if abs(delta_y) > self.cellSize:
            new_delta_y = (abs(delta_y) - self.cellSize) * np.sign(delta_y)
            delta_y = self.cellSize * np.sign(delta_y)
            new_iteration_needed = True
        if abs(delta_x) > self.cellSize:
            new_delta_x = (abs(delta_x) - self.cellSize) * np.sign(delta_x)
            delta_x = self.cellSize * np.sign(delta_x)
            new_iteration_needed = True
        if new_iteration_needed:
            self.translational_motion(new_delta_x, new_delta_y)

        # do transformation
        new_grid = np.zeros(np.shape(self.oLog))
        if delta_x >= 0:
            if delta_y >= 0:
                new_grid[0, :] = self.OZero
                new_grid[:, -1] = self.OZero
                w2 = (self.cellSize - delta_x) * delta_y / self.cellArea
                w3 = delta_x * delta_y / self.cellArea
                w5 = (self.cellSize - delta_x) * (self.cellSize - delta_y) / self.cellArea
                w6 = delta_x * (self.cellSize - delta_y) / self.cellArea
                new_grid[1:, :-1] = w2 * self.oLog[:-1, :-1] + w3 * self.oLog[:-1, 1:] + \
                                    w5 * self.oLog[1:, :-1] + w6 * self.oLog[1:, 1:] + self.OZero

            else:
                new_grid[-1, :] = self.OZero
                new_grid[:, -1] = self.OZero
                w5 = (self.cellSize - delta_x) * (self.cellSize + delta_y) / self.cellArea
                w6 = delta_x * (self.cellSize + delta_y) / self.cellArea
                w8 = (self.cellSize - delta_x) * (-delta_y) / self.cellArea
                w9 = delta_x * (-delta_y) / self.cellArea
                new_grid[:-1, :-1] = w5 * self.oLog[:-1, :-1] + w6 * self.oLog[:-1, 1:] + \
                                     w8 * self.oLog[1:, :-1] + w9 * self.oLog[1:, 1:] + self.OZero
        else:
            if delta_y >= 0:
                new_grid[0, :] = self.OZero
                new_grid[:, 0] = self.OZero
                w1 = -delta_x * delta_y / self.cellArea
                w2 = (self.cellSize + delta_x) * delta_y / self.cellArea
                w4 = -delta_x * (self.cellSize - delta_y) / self.cellArea
                w5 = (self.cellSize + delta_x) * (self.cellSize - delta_y) / self.cellArea
                new_grid[1:, 1:] = w1 * self.oLog[:-1, :-1] + w2 * self.oLog[:-1, 1:] + \
                                   w4 * self.oLog[1:, :-1] + w5 * self.oLog[1:, 1:] + self.OZero
            else:
                new_grid[-1, :] = self.OZero
                new_grid[:, 0] = self.OZero
                w4 = (-delta_x) * (self.cellSize + delta_y) / self.cellArea
                w5 = (self.cellSize + delta_x) * (self.cellSize + delta_y) / self.cellArea
                w7 = (-delta_x) * (-delta_y) / self.cellArea
                w8 = (self.cellSize + delta_x) * (-delta_y) / self.cellArea
                new_grid[:-1, 1:] = w4 * self.oLog[:-1, :-1] + w5 * self.oLog[:-1, 1:] + \
                                    w7 * self.oLog[1:, :-1] + w8 * self.oLog[1:, 1:] + self.OZero
        self.oLog = new_grid

    def cell_rotation(self, delta_psi):
        """
        Rotates the grid
        :param delta_psi: change in heading
        :return: Nothing
        """
        if delta_psi == 0:
            return
        if abs(delta_psi) > self.PI2:
            new_delta_psi = (abs(delta_psi) - self.PI2) * np.sign(delta_psi)
            delta_psi = self.PI2 * np.sign(delta_psi)
            self.rot_motion(new_delta_psi)


        y = .5 * self.cellSize * (1 - math.tan(delta_psi / 2))
        w = 0.25 * self.cellSize * (self.cellSize / (y - self.cellSize) + 2) * y/self.cellArea
        A = 1 - 4*w

        new_grid = np.zeros(np.shape(self.oLog))
        # cells in the middle. A*self + w( up + down + left + right)
        new_grid[1:-1, 1:-1] = A * self.oLog[1:-1, 1:-1] +\
                               w * (self.oLog[:-2, 1:-1] + self.oLog[2:, 1:-1] +
                                    self.oLog[1:-1, :-2] + self.oLog[1:-1, 2:])
        # first row
        new_grid[0, 1:-1] = A * self.oLog[0, 1:-1] +\
                               w * (self.OZero + self.oLog[1, 1:-1] +
                                    self.oLog[0, :-2] + self.oLog[0, 2:])
        # last row
        new_grid[-1, 1:-1] = A * self.oLog[-1, 1:-1] +\
                               w * (self.oLog[-2, 1:-1] + self.OZero +
                                    self.oLog[-1, :-2] + self.oLog[-1, 2:])
        # first column
        new_grid[1:-1, 0] = A * self.oLog[1:-1, 0] +\
                               w * (self.oLog[:-2, 0] + self.oLog[2:, 0] +
                                    self.OZero + self.oLog[1:-1, 2])
        # last column
        new_grid[1:-1, -1] = A * self.oLog[1:-1, -1] +\
                               w * (self.oLog[:-2, -1] + self.oLog[2:, -1] +
                                    self.oLog[1:-1, -2] + self.OZero)
        # upper left
        new_grid[0, 0] = A * self.oLog[0, 0] +\
                               w * (2 * self.OZero + self.oLog[1, 0] + self.oLog[0, 1])
        # upper right
        new_grid[0, -1] = A * self.oLog[0, -1] +\
                               w * (2 * self.OZero + self.oLog[1, 0] + self.oLog[0, -1])
        # lower left
        new_grid[-1, 0] = A * self.oLog[-1, 0] +\
                               w * (self.oLog[-1, 0] + 2 * self.OZero + self.oLog[-1, 1])
        # lower right
        new_grid[-1, -1] = A * self.oLog[-1, -1] + \
                               w * (self.oLog[-2, -1] + 2 * self.OZero + self.oLog[-1, -2])
        self.oLog = new_grid

    def rotate_grid(self, delta_psi):
        # Check if movement is less than MIN_ROT => save for later
        delta_psi += self.old_delta_psi
        self.old_delta_psi = 0
        if abs(delta_psi) < self.MIN_ROT:
            self.old_delta_psi = delta_psi
            return 0

        # Check if movement is to big to rotate fast enough
        if abs(delta_psi) > self.MAX_ROT_BEFORE_RESET:
            self.clearGrid()
            logger.warning('Reset grid because requested rotation was: {:.2f} deg'.format(delta_psi*180/math.pi))
            return 0
        # Check if rotation is to great
        if abs(delta_psi) > self.MAX_ROT:
            n = int(math.floor(abs(delta_psi)/self.MAX_ROT))
            if n > 8:
                logger.error('Stacked rotation is big. n = {}\tDelta_psi_orig={:.2f} deg'.format(n, delta_psi*180/math.pi))

            max_rot_signed = self.MAX_ROT * np.sign(delta_psi)
            delta_psi += -max_rot_signed * n
            for i in range(n):
                self.rotate_grid(max_rot_signed)
        if abs(delta_psi) > self.MAX_ROT:
            logger.error('delta psi > max rot'.format(delta_psi * 180 / math.pi))
        delta_x = self.r*np.sin(delta_psi+self.theta) - self.cell_x_value
        delta_y = self.r*np.cos(delta_psi+self.theta) - self.cell_y_value
        if np.any(np.abs(delta_x) > self.cellSize_with_margin) or np.any(np.abs(delta_y) > self.cellSize_with_margin):
            raise MyException('delta x or y to large')
        new_left_grid = np.ones(np.shape(self.oLog[:, :self.origoJ]), dtype=self.oLog_type)
        new_right_grid = np.ones(np.shape(self.oLog[:, self.origoJ:]), dtype=self.oLog_type)
        if delta_psi >= 0:
            # Left grid both positive. Right grid x_postive, y negative
            new_left_grid[0, :] = self.OZero
            new_left_grid[:, -1] = self.OZero
            wl2 = (self.cellSize - delta_x[:, :self.origoJ+1]) * delta_y[:, :self.origoJ+1] / self.cellArea
            wl3 = delta_x[:, :self.origoJ+1] * delta_y[:, :self.origoJ+1] / self.cellArea
            wl5 = (self.cellSize - delta_x[:, :self.origoJ+1]) * (self.cellSize - delta_y[:, :self.origoJ+1]) / self.cellArea
            wl6 = delta_x[:, :self.origoJ+1] * (self.cellSize - delta_y[:, :self.origoJ+1]) / self.cellArea
            if np.any(np.abs(wl2+wl3+wl5+wl6-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wl2+wl3+wl5+wl6))))
            new_left_grid[1:, :] = wl2[1:, :-1] * self.oLog[:-1, :self.origoJ] + \
                                     wl3[1:, :-1] * self.oLog[:-1, 1:self.origoJ+1] + \
                                     wl5[1:, :-1] * self.oLog[1:, :self.origoJ] + \
                                     wl6[1:, :-1] * self.oLog[1:, 1:self.origoJ+1] + self.OZero

            new_right_grid[-1, :] = self.OZero
            new_right_grid[:, -1] = self.OZero
            wr5 = (self.cellSize - delta_x[:, self.origoJ:]) * (self.cellSize + delta_y[:, self.origoJ:]) / self.cellArea
            wr6 = delta_x[:, self.origoJ:] * (self.cellSize + delta_y[:, self.origoJ:]) / self.cellArea
            wr8 = (self.cellSize - delta_x[:, self.origoJ:]) * (-delta_y[:, self.origoJ:]) / self.cellArea
            wr9 = delta_x[:, self.origoJ:] * (-delta_y[:, self.origoJ:]) / self.cellArea
            if np.any(np.abs(wr5+wr6+wr8+wr9-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wr5+wr6+wr8+wr9))))
            new_right_grid[:-1, :-1] = wr5[:-1, :-1] * self.oLog[:-1, self.origoJ:-1] +\
                                       wr6[:-1, :-1] * self.oLog[:-1, self.origoJ+1:] +\
                                       wr8[:-1, :-1] * self.oLog[1:, self.origoJ:-1] +\
                                       wr9[:-1, :-1] * self.oLog[1:, self.origoJ+1:] + self.OZero
        else:
            # Left grid: both neg, right grid: x neg, y pos
            new_left_grid[-1, :] = self.OZero
            new_left_grid[:, 0] = self.OZero
            wl4 = (-delta_x[:, :self.origoJ]) * (self.cellSize + delta_y[:, :self.origoJ]) / self.cellArea
            wl5 = (self.cellSize + delta_x[:, :self.origoJ]) * (self.cellSize + delta_y[:, :self.origoJ]) / self.cellArea
            wl7 = (-delta_x[:, :self.origoJ]) * (-delta_y[:, :self.origoJ]) / self.cellArea
            wl8 = (self.cellSize + delta_x[:, :self.origoJ]) * (-delta_y[:, :self.origoJ]) / self.cellArea
            if np.any(np.abs(wl4+wl5+wl7+wl8-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wl4+wl5+wl7+wl8))))
            new_left_grid[:-1, 1:] = wl4[1:, :-1] * self.oLog[:-1, :self.origoJ-1] +\
                                     wl5[1:, :-1] * self.oLog[:-1, 1:self.origoJ] +\
                                     wl7[1:, :-1] * self.oLog[1:, :self.origoJ-1] +\
                                     wl8[1:, :-1] * self.oLog[1:, 1:self.origoJ] + self.OZero

            new_right_grid[0, :] = self.OZero
            new_right_grid[:, 0] = self.OZero
            wr1 = -delta_x[:, self.origoJ-1:] * delta_y[:, self.origoJ-1:] / self.cellArea
            wr2 = (self.cellSize + delta_x[:, self.origoJ-1:]) * delta_y[:, self.origoJ-1:] / self.cellArea
            wr4 = -delta_x[:, self.origoJ-1:] * (self.cellSize - delta_y[:, self.origoJ-1:]) / self.cellArea
            wr5 = (self.cellSize + delta_x[:, self.origoJ-1:]) * (self.cellSize - delta_y[:, self.origoJ-1:]) / self.cellArea
            if np.any(np.abs(wr1+wr2+wr4+wr5-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wr1+wr2+wr4+wr5))))
            new_right_grid[1:, :] = wr1[:-1, :-1] * self.oLog[:-1, self.origoJ-1:-1] +\
                                     wr2[:-1, :-1] * self.oLog[:-1, self.origoJ:] +\
                                     wr4[:-1, :-1] * self.oLog[1:, self.origoJ-1:-1] +\
                                     wr5[:-1, :-1] * self.oLog[1:, self.origoJ:] + self.OZero
        self.oLog[:, :self.origoJ] = new_left_grid
        self.oLog[:, self.origoJ:] = new_right_grid
        # self.cell_rotation(delta_psi)

# Exeption class for makin understanable exception
class MyException(Exception):
    pass
