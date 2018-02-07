import numpy as np
import math
import logging
from help import *
import cv2
from settings import BlobDetectorSettings

logger = logging.getLogger('OGrid')


class OGrid(object):
    cur_step = 0
    PI2 = math.pi / 2
    oLog_type = np.float32
    old_delta_x = 0
    old_delta_y = 0
    old_delta_psi = 0
    MIN_ROT = 0.1 * math.pi / 180
    i_max = 1601
    j_max = 1601

    last_bearing = 0
    MAX_BINS = 800
    MAX_CELLS = 4
    RES = 1601
    r_unit = np.zeros((RES, RES))
    theta = np.zeros((RES, RES))
    x_mesh_unit = np.zeros((RES, RES))
    y_mesh_unit = np.zeros((RES, RES))
    map = np.zeros((6400, MAX_BINS, MAX_CELLS), dtype=np.uint32)
    last_data = np.zeros(MAX_BINS, dtype=np.uint8)
    last_distance = 0
    range_scale = 1.0

    def __init__(self, half_grid, p_m, binary_threshold=0.7, cellsize=0):
        if half_grid:
            self.i_max = int((OGrid.RES / 2) * (1 + math.tan(math.pi / 90.0)))
        self.cell_size = cellsize
        self.origin_j = self.origin_i = (OGrid.RES - 1) / 2
        self.OZero = math.log(p_m / (1 - p_m))
        self.binary_threshold = math.log(binary_threshold / (1 - binary_threshold))
        self.o_log = np.ones((self.i_max, self.j_max), dtype=self.oLog_type) * self.OZero
        [self.i_max, self.j_max] = np.shape(self.o_log)
        if not np.any(OGrid.map != 0):
            # self.loadMap()
            with np.load('OGrid_data/map_1601.npz') as data:
                OGrid.map = data['map']
        self.MAX_ROT = math.asin(801/(math.sqrt(2*(800**2))))-math.pi/4
        self.MAX_ROT_BEFORE_RESET = 30 * self.MAX_ROT
        if not np.any(OGrid.r_unit != 0):
            try:
                with np.load('OGrid_data/rad_1601.npz') as data:
                    OGrid.x_mesh_unit = data['x_mesh']
                    OGrid.y_mesh_unit = data['y_mesh']
                    OGrid.r_unit = data['r_unit']
                    OGrid.theta = data['theta']
            except:
                xy_unit = np.linspace(-(OGrid.RES - 1) / 2, (OGrid.RES - 1) / 2, OGrid.RES, True) / OGrid.RES
                OGrid.x_mesh_unit, OGrid.y_mesh_unit = np.meshgrid(xy_unit, xy_unit)
                OGrid.r_unit = np.sqrt(np.power(OGrid.x_mesh_unit, 2) + np.power(OGrid.x_mesh_unit, 2))
                OGrid.theta = np.arctan2(OGrid.y_mesh_unit, OGrid.x_mesh_unit)
                np.savez('OGrid_data/rad_1601.npz', x_mesh=OGrid.x_mesh_unit,
                         y_mesh=OGrid.y_mesh_unit, r=OGrid.r_unit, theta=OGrid.theta)

        # detection
        self.fast_detector = cv2.FastFeatureDetector_create()

        self.blob_detector = cv2.SimpleBlobDetector_create(BlobDetectorSettings.params)

    # def loadMap(self):
    #     binary_file = open('OGrid_data/map_1601_new_no_stride.bin', "rb")
    #     for i in range(0, 6400):
    #         for j in range(0, self.MAX_BINS):
    #             length = (struct.unpack('<B', binary_file.read(1)))[0]
    #             if length > self.MAX_CELLS:
    #                 raise Exception('Map variable to small')
    #             for k in range(0, length):
    #                 OGrid.map[i, j, k] = (struct.unpack('<I', binary_file.read(4)))[0]
    #     binary_file.close()

    def get_p(self):
        try:
            p = 1 - 1 / (1 + np.exp(self.o_log))
        except RuntimeWarning:
            self.o_log[np.nonzero(self.o_log > 50)] = 50
            self.o_log[np.nonzero(self.o_log < -50)] = -50
            p = 1 - 1 / (1 + np.exp(self.o_log))
            logger.debug('Overflow when calculating probability')
        return p

    def get_raw(self):
        return self.o_log

    def get_binary_map(self):
        return (self.o_log > self.binary_threshold).astype(np.float)

    def auto_update_zhou(self, msg, threshold):
        self.range_scale = msg.range_scale
        range_step = self.MAX_BINS / msg.dbytes
        new_data = np.zeros(self.MAX_BINS, dtype=np.uint8)
        updated = np.zeros(self.MAX_BINS, dtype=np.bool)
        try:
            for i in range(0, msg.dbytes):
                new_data[int(round(i * range_step))] = msg.data[i]
                updated[int(round(i * range_step))] = True
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
                    val = (float(new_data[j]) - new_data[i - 1]) / (j - 1 + 1)
                    for k in range(i, j):
                        new_data[k] = val * (k - i + 1) + new_data[i - 1]
                        updated[k] = True
        new_data[np.nonzero(new_data < threshold)] = -self.OZero
        new_data[np.nonzero(new_data > 0)] = 0.5 + self.OZero
        bearing_diff = msg.bearing - self.last_bearing
        beam_half = 27
        if msg.chan2:
            beam_half = 13
        if math.fabs(bearing_diff) <= msg.step:
            if bearing_diff > 0:
                value_gain = (new_data.astype(float) - self.last_data) / bearing_diff
                for n in range(self.last_bearing, msg.bearing + 1):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] += new_data + (n - self.last_bearing) * value_gain
                for n in range(msg.bearing + 1, msg.bearing + beam_half):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] += new_data
            else:
                value_gain = (new_data.astype(float) - self.last_data) / (-bearing_diff)
                for n in range(msg.bearing, self.last_bearing + 1):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] += new_data + (n - msg.bearing) * value_gain
                for n in range(msg.bearing - beam_half, msg.bearing):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] += new_data
        else:
            for n in range(msg.bearing - beam_half, msg.bearing + beam_half):
                for i in range(0, self.MAX_CELLS):
                    self.o_log.flat[OGrid.map[n, :, i]] += new_data
        self.last_bearing = msg.bearing
        self.last_data = new_data

    def update_raw(self, msg):
        self.range_scale = msg.range_scale
        range_step = self.MAX_BINS / msg.dbytes
        new_data = np.zeros(self.MAX_BINS, dtype=np.uint8)
        updated = np.zeros(self.MAX_BINS, dtype=np.bool)
        try:
            for i in range(0, msg.dbytes):
                new_data[int(round(i * range_step))] = msg.data[i]
                updated[int(round(i * range_step))] = True
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
                    val = (float(new_data[j]) - new_data[i - 1]) / (j - 1 + 1)
                    for k in range(i, j):
                        new_data[k] = val * (k - i + 1) + new_data[i - 1]
                        updated[k] = True

        bearing_diff = msg.bearing - self.last_bearing
        beam_half = 27
        if msg.chan2:
            beam_half = 13
        if math.fabs(bearing_diff) <= msg.step:
            if bearing_diff > 0:
                value_gain = (new_data.astype(float) - self.last_data) / bearing_diff
                for n in range(self.last_bearing, msg.bearing + 1):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] = self.last_data + (n - self.last_bearing) * value_gain
                for n in range(msg.bearing + 1, msg.bearing + beam_half):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] = new_data
            else:
                value_gain = (new_data.astype(float) - self.last_data) / (-bearing_diff)
                for n in range(msg.bearing, self.last_bearing + 1):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] = self.last_data + (n - msg.bearing) * value_gain
                for n in range(msg.bearing - beam_half, msg.bearing):
                    for i in range(0, self.MAX_CELLS):
                        self.o_log.flat[OGrid.map[n, :, i]] = new_data
        else:
            for n in range(msg.bearing - beam_half, msg.bearing + beam_half):
                for i in range(0, self.MAX_CELLS):
                    self.o_log.flat[OGrid.map[n, :, i]] = new_data
        self.last_bearing = msg.bearing
        self.last_data = new_data

    def update_distance(self, distance):
        try:
            factor = distance / self.last_distance
        except:
            factor = 1
            self.last_distance = distance
        if factor == 1:
            return
        new_grid = np.ones(shape=np.shape(self.o_log), dtype=self.oLog_type) * self.OZero
        if factor < 1:
            # old distance > new distance
            new_grid = self.o_log[np.meshgrid((np.round((np.arange(0, self.j_max, 1) - self.origin_j) *
                                                        factor + self.origin_j)).astype(dtype=int),
                                              (np.round((np.arange(0, self.i_max, 1) - self.origin_i) *
                                                       factor + self.origin_i)).astype(dtype=int))]
        else:
            # old distance < new distance
            i_lim = int(round(0.5 * self.i_max / factor))
            j_lim = int(round(0.5 * self.j_max / factor))
            new_grid[i_lim:-i_lim, j_lim:-j_lim] = self.o_log[
                np.meshgrid((np.round((np.arange(j_lim, self.j_max - j_lim, 1) - self.origin_j) *
                                      factor + self.origin_j)).astype(dtype=int),
                            (np.round((np.arange(i_lim, self.i_max - i_lim, 1) - self.origin_i) *
                                      factor + self.origin_i)).astype(dtype=int))]
        self.o_log = new_grid
        self.last_distance = distance

    def clear_grid(self):
        self.o_log = np.ones((self.i_max, self.j_max)) * self.OZero
        logger.info('Grid cleared')

    def translational_motion(self, delta_x, delta_y, first):
        """
        transform grid for deterministic translational motion
        :param delta_x: Change in positive grid direction [m] (sway)
        :param delta_y: Change in positive grid direction [m] (surge)
        :return: Nothing
        """
        if first:
            delta_x *= OGrid.MAX_BINS/self.range_scale
            delta_y *= OGrid.MAX_BINS/self.range_scale
        logger.debug('delta_x{}\tdelta_y:{}'.format(delta_x, delta_y))
        # Check if movement is less than 1/4 of cell size => save for later
        delta_x += self.old_delta_x
        delta_y += self.old_delta_y
        if abs(delta_x) < .01:  # self.fourth_cell_size:
            self.old_delta_x = delta_x
            delta_x = 0
        if abs(delta_y) < .01:  # self.fourth_cell_size:
            self.old_delta_y = delta_y
            delta_y = 0
        if delta_y == delta_x == 0:
            return False

        # Check if movement is > cell size => new itteration
        new_iteration_needed = False
        new_delta_y = 0
        new_delta_x = 0
        if abs(delta_y) > 1:
            new_delta_y = (abs(delta_y) - 1) * np.sign(delta_y)
            delta_y = np.sign(delta_y)
            new_iteration_needed = True
        if abs(delta_x) > 1:
            new_delta_x = (abs(delta_x) - 1) * np.sign(delta_x)
            delta_x = np.sign(delta_x)
            new_iteration_needed = True
        if new_iteration_needed:
            self.translational_motion(new_delta_x, new_delta_y, False)

        # do transformation
        new_grid = np.zeros(np.shape(self.o_log))
        if delta_x >= 0:
            if delta_y >= 0:
                w2 = (1 - delta_x) * delta_y
                w3 = delta_x * delta_y
                w5 = (1 - delta_x) * (1 - delta_y)
                w6 = delta_x * (1 - delta_y)
                new_grid[1:, :-1] = w2 * self.o_log[:-1, :-1] + w3 * self.o_log[:-1, 1:] + \
                                    w5 * self.o_log[1:, :-1] + w6 * self.o_log[1:, 1:]

                new_grid[0, :-1] = (w2 + w3) * self.OZero * np.ones(self.j_max - 1) + \
                                   w5 * self.o_log[1, :-1] + w6 * self.o_log[1, 1:]
                new_grid[1:, -1] = w2 * self.o_log[:-1, -2] + (w3 + w6) * self.OZero * np.ones(self.i_max - 1) + \
                                   w5 * self.o_log[1:, -2]
                new_grid[0, -1] = (w2 + w3 + w6) * self.OZero + w5 * self.o_log[1, -2]

            else:
                w5 = (1 - delta_x) * (1 + delta_y)
                w6 = delta_x * (1 + delta_y)
                w8 = (1 - delta_x) * (-delta_y)
                w9 = delta_x * (-delta_y)
                new_grid[:-1, :-1] = w5 * self.o_log[:-1, :-1] + w6 * self.o_log[:-1, 1:] + \
                                     w8 * self.o_log[1:, :-1] + w9 * self.o_log[1:, 1:]

                new_grid[-1, :-1] = w5 * self.o_log[-2, :-1] + w6 * self.o_log[-2, 1:] + \
                                    (w8 + w9) * self.OZero * np.ones(self.j_max - 1)
                new_grid[:-1, -1] = w5 * self.o_log[:-1, -2] + (w6 + w9) * self.OZero * np.ones(self.i_max - 1) + \
                                    w8 * self.o_log[1:, -2]
                new_grid[-1, -1] = w5 * self.o_log[-2, -2] + (w6 + w8 + w9) * self.OZero
        else:
            if delta_y >= 0:
                w1 = -delta_x * delta_y
                w2 = (1 + delta_x) * delta_y
                w4 = -delta_x * (1 - delta_y)
                w5 = (1 + delta_x) * (1 - delta_y)
                new_grid[1:, 1:] = w1 * self.o_log[:-1, :-1] + w2 * self.o_log[:-1, 1:] + \
                                   w4 * self.o_log[1:, :-1] + w5 * self.o_log[1:, 1:]

                new_grid[0, 1:] = (w1 + w2) * self.OZero * np.ones(self.j_max - 1) + \
                                  w4 * self.o_log[0, :-1] + w5 * self.o_log[0, 1:]
                new_grid[1:, 0] = (w1 + w4) * self.OZero * np.ones(self.i_max - 1) + w2 * self.o_log[:-1, 0] + \
                                  w5 * self.o_log[1:, 0]
                new_grid[0, 0] = (w1 + w2 + w4) * self.OZero + w5 * self.o_log[1, 1]
            else:
                w4 = (-delta_x) * (1 + delta_y)
                w5 = (1 + delta_x) * (1 + delta_y)
                w7 = (-delta_x) * (-delta_y)
                w8 = (1 + delta_x) * (-delta_y)
                new_grid[:-1, 1:] = w4 * self.o_log[:-1, :-1] + w5 * self.o_log[:-1, 1:] + \
                                    w7 * self.o_log[1:, :-1] + w8 * self.o_log[1:, 1:]

                new_grid[-1, 1:] = w4 * self.o_log[-2, :-1] + w5 * self.o_log[-2, 1:] + \
                                   (w7 + w8) * self.OZero * np.ones(self.j_max - 1)
                new_grid[:-1, 0] = (w4 + w7) * self.OZero * np.ones(self.i_max - 1) + w5 * self.o_log[:-1, 1] + \
                                   w8 * self.o_log[1:, 1]
                new_grid[-1, 0] = (w4 + w7 + w8) * self.OZero + w5 * self.o_log[-2, 1]
        self.o_log = new_grid
        return True

    def cell_rotation(self, delta_psi):
        """
        Rotates the grid
        :param delta_psi: change in heading
        :return: Nothing
        """
        if delta_psi == 0:
            return False
        if abs(delta_psi) > self.PI2:
            new_delta_psi = (abs(delta_psi) - self.PI2) * np.sign(delta_psi)
            delta_psi = self.PI2 * np.sign(delta_psi)
            self.cell_rotation(new_delta_psi)

        y = .5 * (1 - math.tan(delta_psi / 2))
        w = 0.25 * (1 / (y - 1) + 2) * y
        a = 1 - 4 * w

        new_grid = np.zeros(np.shape(self.o_log))
        # cells in the middle. A*self + w( up + down + left + right)
        new_grid[1:-1, 1:-1] = a * self.o_log[1:-1, 1:-1] + \
                               w * (self.o_log[:-2, 1:-1] + self.o_log[2:, 1:-1] +
                                    self.o_log[1:-1, :-2] + self.o_log[1:-1, 2:])
        # first row
        new_grid[0, 1:-1] = a * self.o_log[0, 1:-1] + \
                            w * (self.OZero + self.o_log[1, 1:-1] +
                                 self.o_log[0, :-2] + self.o_log[0, 2:])
        # last row
        new_grid[-1, 1:-1] = a * self.o_log[-1, 1:-1] + \
                             w * (self.o_log[-2, 1:-1] +
                                  self.o_log[-1, :-2] + self.o_log[-1, 2:])
        # first column
        new_grid[1:-1, 0] = a * self.o_log[1:-1, 0] + \
                            w * (self.o_log[:-2, 0] + self.o_log[2:, 0] +
                                 self.OZero + self.o_log[1:-1, 2])
        # last column
        new_grid[1:-1, -1] = a * self.o_log[1:-1, -1] + \
                             w * (self.o_log[:-2, -1] + self.o_log[2:, -1] +
                                  self.o_log[1:-1, -2] + self.OZero)
        # upper left
        new_grid[0, 0] = a * self.o_log[0, 0] + \
                         w * (2 * self.OZero + self.o_log[1, 0] + self.o_log[0, 1])
        # upper right
        new_grid[0, -1] = a * self.o_log[0, -1] + \
                          w * (2 * self.OZero + self.o_log[1, 0] + self.o_log[0, -1])
        # lower left
        new_grid[-1, 0] = a * self.o_log[-1, 0] + \
                          w * (self.o_log[-1, 0] + 2 * self.OZero + self.o_log[-1, 1])
        # lower right
        new_grid[-1, -1] = a * self.o_log[-1, -1] + \
                           w * (self.o_log[-2, -1] + 2 * self.OZero + self.o_log[-1, -2])
        self.o_log = new_grid
        return True

    def rotate_grid(self, delta_psi):
        # Check if movement is less than MIN_ROT => save for later
        delta_psi += self.old_delta_psi
        self.old_delta_psi = 0
        if abs(delta_psi) < self.MIN_ROT:
            self.old_delta_psi = delta_psi
            return False

        # Check if movement is to big to rotate fast enough
        if abs(delta_psi) > self.MAX_ROT_BEFORE_RESET:
            self.clear_grid()
            logger.warning('Reset grid because requested rotation was: {:.2f} deg'.format(delta_psi * 180 / math.pi))
            return False
        # Check if rotation is to great
        if abs(delta_psi) > self.MAX_ROT:
            n = int(math.floor(abs(delta_psi) / self.MAX_ROT))
            if n > 8:
                logger.error(
                    'Stacked rotation is big. n = {}\tDelta_psi_orig={:.2f} deg'.format(n, delta_psi * 180 / math.pi))

            max_rot_signed = self.MAX_ROT * np.sign(delta_psi)
            delta_psi += -max_rot_signed * n
            for i in range(n):
                self.rotate_grid(max_rot_signed)
        if abs(delta_psi) > self.MAX_ROT:
            logger.error('delta psi > max rot'.format(delta_psi * 180 / math.pi))
        delta_x = self.r_unit * np.sin(delta_psi + OGrid.theta) - self.x_mesh_unit
        delta_y = self.r_unit * np.cos(delta_psi + OGrid.theta) - self.y_mesh_unit
        if np.any(np.abs(delta_x) > 1) or np.any(np.abs(delta_y) > 1):
            print_args(delta_x=delta_x, delta_y=delta_y)
            raise Exception('delta x or y to large')
        new_left_grid = np.zeros(np.shape(self.o_log[:, :self.origin_j]), dtype=self.oLog_type)
        new_right_grid = np.zeros(np.shape(self.o_log[:, self.origin_j:]), dtype=self.oLog_type)
        if delta_psi >= 0:
            # Left grid both positive. Right grid x_postive, y negative
            wl2 = (1 - delta_x[:, :self.origin_j + 1]) * delta_y[:, :self.origin_j + 1]
            wl3 = delta_x[:, :self.origin_j + 1] * delta_y[:, :self.origin_j + 1]
            wl5 = (1 - delta_x[:, :self.origin_j + 1]) * (1 - delta_y[:, :self.origin_j + 1])
            wl6 = delta_x[:, :self.origin_j + 1] * (1 - delta_y[:, :self.origin_j + 1])
            if np.any(np.abs(wl2 + wl3 + wl5 + wl6 - 1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wl2 + wl3 + wl5 + wl6))))
            new_left_grid[1:, :] = wl2[1:, :-1] * self.o_log[:-1, :self.origin_j] + \
                                   wl3[1:, :-1] * self.o_log[:-1, 1:self.origin_j + 1] + \
                                   wl5[1:, :-1] * self.o_log[1:, :self.origin_j] + \
                                   wl6[1:, :-1] * self.o_log[1:, 1:self.origin_j + 1]

            new_left_grid[0, :] = (wl2[0, :-1] + wl3[0, :-1]) * self.OZero * np.ones(np.shape(new_left_grid[0, :])) + \
                                  wl5[0, :-1] * self.o_log[1, :self.origin_j] + \
                                  wl6[0, :-1] * self.o_log[1, 1:self.origin_j + 1]

            wr5 = (1 - delta_x[:, self.origin_j:]) * (1 + delta_y[:, self.origin_j:])
            wr6 = delta_x[:, self.origin_j:] * (1 + delta_y[:, self.origin_j:])
            wr8 = (1 - delta_x[:, self.origin_j:]) * (-delta_y[:, self.origin_j:])
            wr9 = delta_x[:, self.origin_j:] * (-delta_y[:, self.origin_j:])
            if np.any(np.abs(wr5 + wr6 + wr8 + wr9 - 1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wr5 + wr6 + wr8 + wr9))))
            new_right_grid[:-1, :-1] = wr5[:-1, :-1] * self.o_log[:-1, self.origin_j:-1] + \
                                       wr6[:-1, :-1] * self.o_log[:-1, self.origin_j + 1:] + \
                                       wr8[:-1, :-1] * self.o_log[1:, self.origin_j:-1] + \
                                       wr9[:-1, :-1] * self.o_log[1:, self.origin_j + 1:]

            new_right_grid[-1, :-1] = wr5[-1, :-1] * self.o_log[-1, self.origin_j:-1] + \
                                      wr6[-1, :-1] * self.o_log[-1, self.origin_j + 1:] + \
                                      (wr8[-1, :-1] + wr9[-1, :-1]) * self.OZero * np.ones(
                np.shape(new_right_grid[-1, :-1]))
            new_right_grid[:-1, -1] = wr5[:-1, -1] * self.o_log[:-1, -1] + \
                                      wr8[:-1, -1] * self.o_log[1:, -1] + \
                                      (wr6[:-1, -1] + wr9[:-1, -1]) * self.OZero * np.ones(
                np.shape(new_right_grid[:-1, -1]))
            new_right_grid[-1, -1] = wr5[-1, -1] * self.o_log[-1, -1] + \
                                     (wr6[-1, -1] + wr8[-1, -1] + wr9[-1, -1]) * self.OZero
        else:
            # Left grid: both neg, right grid: x neg, y pos
            wl4 = (-delta_x[:, :self.origin_j]) * (1 + delta_y[:, :self.origin_j])
            wl5 = (1 + delta_x[:, :self.origin_j]) * (1 + delta_y[:, :self.origin_j])
            wl7 = (-delta_x[:, :self.origin_j]) * (-delta_y[:, :self.origin_j])
            wl8 = (1 + delta_x[:, :self.origin_j]) * (-delta_y[:, :self.origin_j])
            if np.any(np.abs(wl4 + wl5 + wl7 + wl8 - 1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wl4 + wl5 + wl7 + wl8))))
            new_left_grid[:-1, 1:] = wl4[1:, :-1] * self.o_log[:-1, :self.origin_j - 1] + \
                                     wl5[1:, :-1] * self.o_log[:-1, 1:self.origin_j] + \
                                     wl7[1:, :-1] * self.o_log[1:, :self.origin_j - 1] + \
                                     wl8[1:, :-1] * self.o_log[1:, 1:self.origin_j]

            new_left_grid[-1, :-1] = wl4[-1, :-1] * self.o_log[-1, :self.origin_j - 1] + \
                                     wl5[-1, :-1] * self.o_log[-1, 1:self.origin_j] + \
                                     (wl7[-1, :-1] + wl8[-1, :-1]) * self.OZero * np.ones(
                np.shape(new_left_grid[-1, :-1]))
            new_left_grid[1:, 0] = wl5[1:, 0] * self.o_log[:-1, 1] + \
                                   (wl4[1:, 0] + wl7[1:, 0]) * self.OZero * np.ones(np.shape(new_left_grid[1:, 0])) + \
                                   wl8[1:, 0] * self.o_log[1:, 1]
            new_left_grid[-1, 0] = wl5[-1, 0] * self.o_log[-1, 0] + \
                                   (wl4[-1, 0] + wl7[-1, 0] + wl8[-1, 0]) * self.OZero

            wr1 = -delta_x[:, self.origin_j - 1:] * delta_y[:, self.origin_j - 1:]
            wr2 = (1 + delta_x[:, self.origin_j - 1:]) * delta_y[:, self.origin_j - 1:]
            wr4 = -delta_x[:, self.origin_j - 1:] * (1 - delta_y[:, self.origin_j - 1:])
            wr5 = (1 + delta_x[:, self.origin_j - 1:]) * (1 - delta_y[:, self.origin_j - 1:])
            if np.any(np.abs(wr1 + wr2 + wr4 + wr5 - 1) > 0.00001):
                logger.debug('Sum = {}!'.format(np.max(np.max(wr1 + wr2 + wr4 + wr5))))
            new_right_grid[1:, :] = wr1[:-1, :-1] * self.o_log[:-1, self.origin_j - 1:-1] + \
                                    wr2[:-1, :-1] * self.o_log[:-1, self.origin_j:] + \
                                    wr4[:-1, :-1] * self.o_log[1:, self.origin_j - 1:-1] + \
                                    wr5[:-1, :-1] * self.o_log[1:, self.origin_j:]
            new_right_grid[0, :] = (wr1[0, 1:] + wr2[0, 1:]) * self.OZero * np.ones(np.shape(new_right_grid[0, :])) + \
                                   wr4[0, 1:] * self.o_log[0, self.origin_j - 1:-1] + \
                                   wr5[0, 1:] * self.o_log[0, self.origin_j:]
        self.o_log[:, :self.origin_j] = new_left_grid
        self.o_log[:, self.origin_j:] = new_right_grid
        # self.cell_rotation(delta_psi)
        return True

    def get_obstacles_fast(self, threshold):
        self.fast_detector.setThreshold(threshold)
        return cv2.cvtColor(cv2.drawKeypoints(cv2.applyColorMap(self.o_log.astype(np.uint8), cv2.COLORMAP_HOT),
                                              self.fast_detector.detect(self.o_log.astype(np.uint8)),
                                              np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
                            cv2.COLOR_BGR2RGB)


    def get_obstacles_blob(self, threshold):
        ret, thresh = cv2.threshold(self.o_log.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(cv2.drawKeypoints(self.o_log.astype(np.uint8),
                                              self.blob_detector.detect(thresh),
                                              np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
                            cv2.COLOR_BGR2RGB)

    def get_obstacles_fast_separation(self, lim):
        keypoints = self.fast_detector.detect(self.o_log.astype(np.uint8))
        if len(keypoints) > 1:
            L = len(keypoints)
            x = np.zeros(L)
            y = np.zeros(L)
            counter = 0
            map = np.zeros(L, dtype=np.uint8)
            for i in range(0, L):
                x[i] = keypoints[i].pt[1]
                y[i] = keypoints[i].pt[0]
            for i in range(0, L):
                x2 = np.power((x[i] - x), 2)
                y2 = np.power((y[i] - y), 2)
                r = np.sqrt(x2 + y2) < lim
                if map[i] != 0:
                    map[r] = map[i]
                else:
                    counter += 1
                    map[r] = counter

            labels = [[] for i in range(np.argmax(map))]

            for i in range(0, L):
                labels[map[i]].append(keypoints[i])

            im_with_keypoints = cv2.applyColorMap(self.o_log.astype(np.uint8), cv2.COLORMAP_HOT)
            for keypoints in labels:
                if len(keypoints) > 1:
                    R = np.random.randint(0, 255)
                    G = np.random.randint(0, 255)
                    B = np.random.randint(0, 255)
                    im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (R, G, B),
                                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return cv2.cvtColor(im_with_keypoints,cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(cv2.applyColorMap(self.o_log.astype(np.uint8), cv2.COLORMAP_HOT),cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    grid = OGrid(True, 0.7)
    grid.o_log = np.load('test.npz')['olog']
    plt.imshow(grid.o_log)
    plt.show()
    grid.translational_motion(0, 10, 30)