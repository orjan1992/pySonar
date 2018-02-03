import struct

import numpy as np
import math
from pathlib import Path
import logging
import os
from scipy.interpolate import *


logger = logging.getLogger('OGrid')


class OGrid(object):
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
    MIN_ROT = 0.1*math.pi/180
    i_max = 1601
    j_max = 1601

    last_bearing = 0
    MAX_BINS = 800
    MAX_CELLS = 4
    map = np.zeros((6400, MAX_BINS, MAX_CELLS), dtype=np.uint32)
    last_data = np.zeros(MAX_BINS, dtype=np.uint8)
    current_distance = 0

    def __init__(self, half_grid, p_m, binary_threshold=0.7):
            # TODO: assignment and usage of static variables. OGrid.test_variable vs self.test_variable
            self.j_maxLimMeters = 10
            if half_grid:
                self.i_maxLimMeters = 5
                self.i_max = int((1601 / 2) * (1 + math.tan(math.pi / 90.0)))
            else:
                self.i_maxLimMeters = self.j_maxLimMeters
            self.cellSize = self.j_maxLimMeters/800
            self.fourth_cell_size = self.cellSize/4
            self.half_cell_size = self.cellSize/2
            self.cellArea = self.cellSize ** 2
            self.origoJ = 800
            self.origoI = 800
            self.OZero = math.log(p_m / (1 - p_m))
            self.binary_threshold = math.log(binary_threshold/(1-binary_threshold))
            self.oLog = np.ones((self.i_max, self.j_max), dtype=self.oLog_type) * self.OZero
            self.O_logic = np.zeros((self.i_max, self.j_max), dtype=bool)
            [self.iMax, self.jMax] = np.shape(self.oLog)
            if not np.any(self.map != 0):
                # self.loadMap()
                self.map = np.load('OGrid_data/map_1601.npz')['map']

    def loadMap(self):
        binary_file = open('OGrid_data/map_1601_new_no_stride.bin', "rb")
        for i in range(0, 6400):
            for j in range(0, self.MAX_BINS):
                length = (struct.unpack('<B', binary_file.read(1)))[0]
                if length > self.MAX_CELLS:
                    raise Exception('Map variable to small')
                for k in range(0, length):
                    self.map[i, j, k] = (struct.unpack('<I', binary_file.read(4)))[0]
        binary_file.close()

    def get_p(self):
        try:
            P = 1 - 1 / (1 + np.exp(self.oLog))
        except RuntimeWarning:
            self.oLog[np.nonzero(self.oLog > 50)] = 50
            self.oLog[np.nonzero(self.oLog < -50)] = -50
            P = 1 - 1 / (1 + np.exp(self.oLog))
            logger.debug('Overflow when calculating probability')
        return P

    def get_raw(self):
        return self.oLog

    def get_binary_map(self):
        return (self.oLog > self.binary_threshold).astype(np.float)

    def updateCellsZhou2(self, cone, rangeScale, theta):
        # UPDATECELLSZHOU
        subRange = cone[self.rHigh.flat[cone] < rangeScale - self.deltaSurface]
        onRange = cone[self.rLow.flat[cone] < (rangeScale + self.deltaSurface)]
        onRange = onRange[self.rHigh.flat[onRange] > (rangeScale - self.deltaSurface)]

        self.oLog.flat[subRange] -= self.OZero  #2.19722  #self.OZero  # 4.595119850134590

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
        range_step = self.MAX_BINS / msg.dbytes
        new_data = np.zeros(self.MAX_BINS, dtype=np.uint8)
        updated = np.zeros(self.MAX_BINS, dtype=np.bool)
        try:
            for i in range(0, msg.dbytes):
                new_data[int(round(i*range_step))] = msg.data[i]
                updated[int(round(i*range_step))] = True
        except Exception as e:
            logger.debug('Mapping to unibins: {0}'.format(e))
        new_data[np.nonzero(new_data < threshold)] = 0
        for i in range(0, self.MAX_BINS):
            if not updated[i]:
                j = i + 1
                while j < self.MAX_BINS:
                    if updated[j]:
                        break
                    j += 1

                if j < self.MAX_BINS:
                    val = (float(new_data[j]) - new_data[i-1])/(j-1+1)
                    for k in range(i, j):
                        new_data[k] = val*(k-i+1) + new_data[i-1]
                        updated[k] = True
        new_data[np.nonzero(new_data < threshold)] = -self.OZero
        new_data[np.nonzero(new_data > 0)] = 0.5 + self.OZero
        bearing_diff = msg.bearing - self.last_bearing
        beam_half = 27
        if msg.chan2:
            beam_half = 13
        if math.fabs(bearing_diff) <= msg.step:
            if bearing_diff > 0:
                value_gain = (new_data.astype(float) - self.last_data)/bearing_diff
                for n in range(self.last_bearing, msg.bearing+1):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] += new_data + (n-self.last_bearing)*value_gain
                for n in range(msg.bearing+1, msg.bearing + beam_half):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] += new_data
            else:
                value_gain = (new_data.astype(float) - self.last_data)/(-bearing_diff)
                for n in range(msg.bearing, self.last_bearing+1):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] += new_data + (n-msg.bearing)*value_gain
                for n in range(msg.bearing- beam_half, msg.bearing):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] += new_data
        else:
            for n in range(msg.bearing - beam_half, msg.bearing + beam_half):
                for i in range(0, self.MAX_CELLS):
                    self.oLog.flat[self.map[n, :, i]] += new_data
        self.last_bearing = msg.bearing
        self.last_data = new_data

    def update_raw(self, msg):
        range_step = self.MAX_BINS / msg.dbytes
        new_data = np.zeros(self.MAX_BINS, dtype=np.uint8)
        updated = np.zeros(self.MAX_BINS, dtype=np.bool)
        try:
            for i in range(0, msg.dbytes):
                new_data[int(round(i*range_step))] = msg.data[i]
                updated[int(round(i*range_step))] = True
        except Exception as e:
            logger.debug('Mapping to unibins: {0}'.format(e))
        for i in range(0, self.MAX_BINS):
            if not updated[i]:
                j = i + 1
                while j < self.MAX_BINS:
                    if updated[j]:
                        break
                    j += 1

                if j < self.MAX_BINS:
                    val = (float(new_data[j]) - new_data[i-1])/(j-1+1)
                    for k in range(i, j):
                        new_data[k] = val*(k-i+1) + new_data[i-1]
                        updated[k] = True

        bearing_diff = msg.bearing - self.last_bearing
        beam_half = 27
        if msg.chan2:
            beam_half = 13
        if math.fabs(bearing_diff) <= msg.step:
            if bearing_diff > 0:
                value_gain = (new_data.astype(float) - self.last_data)/bearing_diff
                for n in range(self.last_bearing, msg.bearing+1):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] = new_data + (n-self.last_bearing)*value_gain
                for n in range(msg.bearing+1, msg.bearing + beam_half):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] = new_data
            else:
                value_gain = (new_data.astype(float) - self.last_data)/(-bearing_diff)
                for n in range(msg.bearing, self.last_bearing+1):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] = new_data + (n-msg.bearing)*value_gain
                for n in range(msg.bearing - beam_half, msg.bearing):
                    for i in range(0, self.MAX_CELLS):
                        self.oLog.flat[self.map[n, :, i]] = new_data
        else:
            for n in range(msg.bearing - beam_half, msg.bearing + beam_half):
                for i in range(0, self.MAX_CELLS):
                    self.oLog.flat[self.map[n, :, i]] = new_data
        self.last_bearing = msg.bearing
        self.last_data = new_data

    def update_distance(self, distance):
        factor = distance / self.current_distance
        if factor == 1:
            return
        new_grid = np.zeros(shape=np.shape(self.oLog), dtype=self.oLog_type)
        if factor < 1:
            # old distance > new distance
            for i in range(0, self.iMax):
                for j in range(0, self.jMax):
                    new_grid[i, j] = self.oLog[round((i - self.origoI)*factor)+self.origoI,
                                               round((j - self.origoJ)*factor)+self.origoJ]
            # TODO: Finish this. Check Scipy interpolation, probably faster. scipy.interpolate.RectBivariateSpline

        else:
            # old distance < new distance
            # TODO: Finish this
            raise NotImplementedError
        self.current_distance = distance

    def clearGrid(self):
        self.oLog = np.ones((self.i_max, self.j_max)) * self.OZero
        self.O_logic = np.zeros((self.i_max, self.j_max), dtype=bool)
        logger.info('Grid cleared')

    def translational_motion(self, delta_x, delta_y):
        """
        transform grid for deterministic translational motion
        :param delta_x: Change in positive grid direction [m] (sway)
        :param delta_y: Change in positive grid direction [m] (surge)
        :return: Nothing
        """
        logger.debug('delta_x{}\tdelta_y:{}'.format(delta_x, delta_y))
        # Check if movement is less than 1/4 of cell size => save for later
        delta_x += self.old_delta_x
        delta_y += self.old_delta_y
        if abs(delta_x) < .01:  #self.fourth_cell_size:
            self.old_delta_x = delta_x
            delta_x = 0
        if abs(delta_y) < .01:  #self.fourth_cell_size:
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
                w2 = (self.cellSize - delta_x) * delta_y / self.cellArea
                w3 = delta_x * delta_y / self.cellArea
                w5 = (self.cellSize - delta_x) * (self.cellSize - delta_y) / self.cellArea
                w6 = delta_x * (self.cellSize - delta_y) / self.cellArea
                new_grid[1:, :-1] = w2 * self.oLog[:-1, :-1] + w3 * self.oLog[:-1, 1:] + \
                                    w5 * self.oLog[1:, :-1] + w6 * self.oLog[1:, 1:]

                new_grid[0, :-1] = (w2+w3) * self.OZero*np.ones(self.jMax-1) + \
                                    w5 * self.oLog[1, :-1] + w6 * self.oLog[1, 1:]
                new_grid[1:, -1] = w2 * self.oLog[:-1, -2] + (w3+w6) * self.OZero*np.ones(self.iMax-1) + \
                                    w5 * self.oLog[1:, -2]
                new_grid[0, -1] = (w2+w3+w6)*self.OZero + w5*self.oLog[1, -2]

            else:
                w5 = (self.cellSize - delta_x) * (self.cellSize + delta_y) / self.cellArea
                w6 = delta_x * (self.cellSize + delta_y) / self.cellArea
                w8 = (self.cellSize - delta_x) * (-delta_y) / self.cellArea
                w9 = delta_x * (-delta_y) / self.cellArea
                new_grid[:-1, :-1] = w5 * self.oLog[:-1, :-1] + w6 * self.oLog[:-1, 1:] + \
                                     w8 * self.oLog[1:, :-1] + w9 * self.oLog[1:, 1:]

                new_grid[-1, :-1] = w5 * self.oLog[-2, :-1] + w6 * self.oLog[-2, 1:] + \
                                  (w8 + w9) * self.OZero*np.ones(self.jMax-1)
                new_grid[:-1, -1] = w5 * self.oLog[:-1, -2] + (w6 + w9) * self.OZero*np.ones(self.iMax-1) + \
                                     w8 * self.oLog[1:, -2]
                new_grid[-1, -1] = w5 * self.oLog[-2, -2] + (w6+w8+w9) * self.OZero
        else:
            if delta_y >= 0:
                w1 = -delta_x * delta_y / self.cellArea
                w2 = (self.cellSize + delta_x) * delta_y / self.cellArea
                w4 = -delta_x * (self.cellSize - delta_y) / self.cellArea
                w5 = (self.cellSize + delta_x) * (self.cellSize - delta_y) / self.cellArea
                new_grid[1:, 1:] = w1 * self.oLog[:-1, :-1] + w2 * self.oLog[:-1, 1:] + \
                                   w4 * self.oLog[1:, :-1] + w5 * self.oLog[1:, 1:]

                new_grid[0, 1:] = (w1 + w2) * self.OZero*np.ones(self.jMax-1) + \
                                   w4 * self.oLog[0, :-1] + w5 * self.oLog[0, 1:]
                new_grid[1:, 0] = (w1 + w4) * self.OZero*np.ones(self.iMax-1) + w2 * self.oLog[:-1, 0] + \
                                   w5 * self.oLog[1:, 0]
                new_grid[0, 0] = (w1 + w2 + w4) * self.OZero + w5 * self.oLog[1, 1]
            else:
                w4 = (-delta_x) * (self.cellSize + delta_y) / self.cellArea
                w5 = (self.cellSize + delta_x) * (self.cellSize + delta_y) / self.cellArea
                w7 = (-delta_x) * (-delta_y) / self.cellArea
                w8 = (self.cellSize + delta_x) * (-delta_y) / self.cellArea
                new_grid[:-1, 1:] = w4 * self.oLog[:-1, :-1] + w5 * self.oLog[:-1, 1:] + \
                                    w7 * self.oLog[1:, :-1] + w8 * self.oLog[1:, 1:]

                new_grid[-1, 1:] = w4 * self.oLog[-2, :-1] + w5 * self.oLog[-2, 1:] + \
                                   (w7 + w8) * self.OZero*np.ones(self.jMax-1)
                new_grid[:-1, 0] = (w4+w7)*self.OZero*np.ones(self.iMax-1) + w5 * self.oLog[:-1, 1] + \
                                    w8 * self.oLog[1:, 1]
                new_grid[-1, 0] = (w4+w7+w8)*self.OZero + w5 * self.oLog[-2, 1]
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
                               w * (self.oLog[-2, 1:-1] +
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
        new_left_grid = np.zeros(np.shape(self.oLog[:, :self.origoJ]), dtype=self.oLog_type)
        new_right_grid = np.zeros(np.shape(self.oLog[:, self.origoJ:]), dtype=self.oLog_type)
        if delta_psi >= 0:
            # Left grid both positive. Right grid x_postive, y negative
            wl2 = (self.cellSize - delta_x[:, :self.origoJ+1]) * delta_y[:, :self.origoJ+1] / self.cellArea
            wl3 = delta_x[:, :self.origoJ+1] * delta_y[:, :self.origoJ+1] / self.cellArea
            wl5 = (self.cellSize - delta_x[:, :self.origoJ+1]) * (self.cellSize - delta_y[:, :self.origoJ+1]) / self.cellArea
            wl6 = delta_x[:, :self.origoJ+1] * (self.cellSize - delta_y[:, :self.origoJ+1]) / self.cellArea
            if np.any(np.abs(wl2+wl3+wl5+wl6-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wl2+wl3+wl5+wl6))))
            new_left_grid[1:, :] = wl2[1:, :-1] * self.oLog[:-1, :self.origoJ] + \
                                     wl3[1:, :-1] * self.oLog[:-1, 1:self.origoJ+1] + \
                                     wl5[1:, :-1] * self.oLog[1:, :self.origoJ] + \
                                     wl6[1:, :-1] * self.oLog[1:, 1:self.origoJ+1]

            new_left_grid[0, :] = (wl2[0, :-1]+wl3[0, :-1]) * self.OZero*np.ones(np.shape(new_left_grid[0, :])) + \
                                     wl5[0, :-1] * self.oLog[1, :self.origoJ] + \
                                     wl6[0, :-1] * self.oLog[1, 1:self.origoJ+1]

            wr5 = (self.cellSize - delta_x[:, self.origoJ:]) * (self.cellSize + delta_y[:, self.origoJ:]) / self.cellArea
            wr6 = delta_x[:, self.origoJ:] * (self.cellSize + delta_y[:, self.origoJ:]) / self.cellArea
            wr8 = (self.cellSize - delta_x[:, self.origoJ:]) * (-delta_y[:, self.origoJ:]) / self.cellArea
            wr9 = delta_x[:, self.origoJ:] * (-delta_y[:, self.origoJ:]) / self.cellArea
            if np.any(np.abs(wr5+wr6+wr8+wr9-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wr5+wr6+wr8+wr9))))
            new_right_grid[:-1, :-1] = wr5[:-1, :-1] * self.oLog[:-1, self.origoJ:-1] +\
                                       wr6[:-1, :-1] * self.oLog[:-1, self.origoJ+1:] +\
                                       wr8[:-1, :-1] * self.oLog[1:, self.origoJ:-1] +\
                                       wr9[:-1, :-1] * self.oLog[1:, self.origoJ+1:]

            new_right_grid[-1, :-1] = wr5[-1, :-1] * self.oLog[-1, self.origoJ:-1] +\
                                    wr6[-1, :-1] * self.oLog[-1, self.origoJ+1:] +\
                                    (wr8[-1, :-1] + wr9[-1, :-1]) * self.OZero*np.ones(np.shape(new_right_grid[-1, :-1]))
            new_right_grid[:-1, -1] = wr5[:-1, -1] * self.oLog[:-1, -1] +\
                                    wr8[:-1, -1] * self.oLog[1:, -1] +\
                                    (wr6[:-1, -1] + wr9[:-1, -1]) * self.OZero*np.ones(np.shape(new_right_grid[:-1, -1]))
            new_right_grid[-1, -1] = wr5[-1, -1] * self.oLog[-1, -1] +\
                                    (wr6[-1, -1] + wr8[-1, -1] + wr9[-1, -1]) * self.OZero
        else:
            # Left grid: both neg, right grid: x neg, y pos
            wl4 = (-delta_x[:, :self.origoJ]) * (self.cellSize + delta_y[:, :self.origoJ]) / self.cellArea
            wl5 = (self.cellSize + delta_x[:, :self.origoJ]) * (self.cellSize + delta_y[:, :self.origoJ]) / self.cellArea
            wl7 = (-delta_x[:, :self.origoJ]) * (-delta_y[:, :self.origoJ]) / self.cellArea
            wl8 = (self.cellSize + delta_x[:, :self.origoJ]) * (-delta_y[:, :self.origoJ]) / self.cellArea
            if np.any(np.abs(wl4+wl5+wl7+wl8-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wl4+wl5+wl7+wl8))))
            new_left_grid[:-1, 1:] = wl4[1:, :-1] * self.oLog[:-1, :self.origoJ-1] +\
                                     wl5[1:, :-1] * self.oLog[:-1, 1:self.origoJ] +\
                                     wl7[1:, :-1] * self.oLog[1:, :self.origoJ-1] +\
                                     wl8[1:, :-1] * self.oLog[1:, 1:self.origoJ]

            new_left_grid[-1, :-1] = wl4[-1, :-1] * self.oLog[-1, :self.origoJ-1] +\
                                     wl5[-1, :-1] * self.oLog[-1, 1:self.origoJ] +\
                                   (wl7[-1, :-1] + wl8[-1, :-1]) * self.OZero*np.ones(np.shape(new_left_grid[-1, :-1]))
            new_left_grid[1:, 0] = wl5[1:, 0] * self.oLog[:-1, 1] +\
                                     (wl4[1:, 0] + wl7[1:, 0]) * self.OZero*np.ones(np.shape(new_left_grid[1:, 0])) +\
                                     wl8[1:, 0] * self.oLog[1:, 1]
            new_left_grid[-1, 0] = wl5[-1, 0] * self.oLog[-1, 0] +\
                                     (wl4[-1, 0] + wl7[-1, 0] + wl8[-1, 0]) * self.OZero

            wr1 = -delta_x[:, self.origoJ-1:] * delta_y[:, self.origoJ-1:] / self.cellArea
            wr2 = (self.cellSize + delta_x[:, self.origoJ-1:]) * delta_y[:, self.origoJ-1:] / self.cellArea
            wr4 = -delta_x[:, self.origoJ-1:] * (self.cellSize - delta_y[:, self.origoJ-1:]) / self.cellArea
            wr5 = (self.cellSize + delta_x[:, self.origoJ-1:]) * (self.cellSize - delta_y[:, self.origoJ-1:]) / self.cellArea
            if np.any(np.abs(wr1+wr2+wr4+wr5-1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wr1+wr2+wr4+wr5))))
            new_right_grid[1:, :] = wr1[:-1, :-1] * self.oLog[:-1, self.origoJ-1:-1] +\
                                     wr2[:-1, :-1] * self.oLog[:-1, self.origoJ:] +\
                                     wr4[:-1, :-1] * self.oLog[1:, self.origoJ-1:-1] +\
                                     wr5[:-1, :-1] * self.oLog[1:, self.origoJ:]
            new_right_grid[0, :] = (wr1[0, 1:]+wr2[0, 1:]) * self.OZero*np.ones(np.shape(new_right_grid[0, :])) +\
                                     wr4[0, 1:] * self.oLog[0, self.origoJ-1:-1] +\
                                     wr5[0, 1:] * self.oLog[0, self.origoJ:]
        self.oLog[:, :self.origoJ] = new_left_grid
        self.oLog[:, self.origoJ:] = new_right_grid
        # self.cell_rotation(delta_psi)

# Exeption class for makin understanable exception
class MyException(Exception):
    pass
