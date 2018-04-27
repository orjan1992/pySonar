from ogrid.rawGrid import RawGrid
import numpy as np
from scipy.signal import argrelextrema
from settings import *
from coordinate_transformations import wrapTo2Pi, wrapToPi, wrapToPiHalf, grid2vehicle_rad
from threading import Lock
import logging
logger = logging.getLogger('OccupancyGrid')

class OccupancyGrid(RawGrid):
    # counter = None
    # sign = None
    contours = None
    dummy = 1
    bin_map = np.zeros((RawGrid.i_max, RawGrid.j_max), dtype=np.uint8)

    def __init__(self, half_grid, p_zero, p_occ, p_free, p_bin_threshold, cell_factor):
        self.p_log_threshold = np.log(p_bin_threshold / (1 - p_bin_threshold))
        self.p_log_zero = np.log(p_zero / (1 - p_zero))
        self.p_log_occ = np.log(p_occ / (1 - p_occ))
        self.p_log_free = np.log(p_free / (1 - p_free))
        super().__init__(half_grid, self.p_log_zero)
        self.im = np.zeros((self.RES, self.RES, 3))
        self.reliable = True
        self.cell_factor = cell_factor
        self.size = int((self.RES - 1) // self.cell_factor)
        self.kernel = np.ones((cell_factor, cell_factor), dtype=np.uint8)
        self.occ2raw_matrix = np.ones((cell_factor, cell_factor))
        try:
            with np.load('ogrid/OGrid_data/occ_map_{}.npz'.format(int(cell_factor))) as data:
                self.angle2cell_low = data['angle2cell_low']
                self.angle2cell_rad_low = data['angle2cell_rad_low']
                self.angle2cell_high = data['angle2cell_high']
                self.angle2cell_rad_high = data['angle2cell_rad_high']
                # TODO: Probably something wrong with half of these angles
                self.occ_map_theta = data['angles']
                self.low_indexer = data['low_indexer']
                self.high_indexer = data['high_indexer']
        except Exception as e:
            self.calc_occ_map(cell_factor)

        self.contour_lock = Lock()

    def occ2raw(self, occ_grid):
        nonzero = np.nonzero(occ_grid)
        if len(nonzero[0]) > 1:
            y1 = np.min(nonzero[0])
            y2 = np.max(nonzero[0]) + 1
            x1 = np.min(nonzero[1])
            x2 = np.max(nonzero[1]) + 1
            self.grid[y1 * self.cell_factor:y2 * self.cell_factor, x1 * self.cell_factor:x2 * self.cell_factor] += np.kron(
                occ_grid[y1:y2, x1:x2], self.occ2raw_matrix)

    def raw2occ(self):
        occ_grid = np.ones((self.size, self.size), dtype=self.oLog_type)
        for i in range(self.size):
            for j in range(self.size):
                occ_grid[i, j] = np.mean(self.grid[i:i+self.cell_factor, j:j+self.cell_factor])
        return occ_grid

    def calc_occ_map(self, factor):
        if factor % 2 != 0:
            raise ValueError('Wrong size reduction')
        size = int((self.RES - 1) / factor)
        size_half = size // 2
        f2 = factor // 2
        range_map = self.r_unit[np.meshgrid(np.arange(factor/2, 1601-factor/2, factor, np.int), np.arange(factor/2, 1601-factor/2, factor, np.int))]*801

        GRAD2RAD = np.pi / 3200.0
        def get_range(ind):
            return range_map.flat[ind]

        # high_indices = [[] for x in range(self.N_ANGLE_STEPS)]
        # low_indices = [[] for x in range(self.N_ANGLE_STEPS)]
        # high_ranges = [[] for x in range(self.N_ANGLE_STEPS)]
        # low_ranges = [[] for x in range(self.N_ANGLE_STEPS)]
        high_indices = []
        high_ranges = []
        high_indexer = np.zeros(self.N_ANGLE_STEPS, dtype=np.int16)
        high_last_indices = np.zeros(1)
        low_indices = []
        low_ranges = []
        low_indexer = np.zeros(self.N_ANGLE_STEPS, dtype=np.int16)
        low_last_indices = np.zeros(1)
        for i in range(self.N_ANGLE_STEPS):
            print('\t' + str(i))
            i_min_high_freq = i - 13
            i_max_high_freq = i + 13
            i_min_high_freq_rad = i_min_high_freq*GRAD2RAD
            i_max_high_freq_rad = i_max_high_freq*GRAD2RAD
            p1 = np.round([size_half, size_half]).astype(int)
            p2 = np.round([size_half*(1-np.sin(i_min_high_freq_rad)), size_half*(1+np.cos(i_min_high_freq_rad))]).astype(int)
            p3 = np.round([size_half*(1-np.sin(i*GRAD2RAD)), size_half*(1+np.cos(i*GRAD2RAD))]).astype(int)
            p4 = np.round([size_half*(1-np.sin(i_max_high_freq_rad)), size_half*(1+np.cos(i_max_high_freq_rad))]).astype(int)
            pts = np.array([p1, p2, p3, p4], dtype=np.int32)
            bin_map = cv2.drawContours(np.zeros((size, size), dtype=np.uint8), [pts], -1, (255, 255, 255), -1)
            indices = np.ravel_multi_index(np.nonzero(bin_map), (size, size))
            if np.any(indices != high_last_indices):
                high_indices.append(np.array(sorted(indices, key=get_range), dtype=np.int32))
                high_ranges.append(range_map.flat[high_indices[-1]])
                high_indexer[i] = len(high_indices)-1

                high_last_indices = indices
            else:
                high_indexer[i] = high_indexer[-1]

            i_min_low_freq = i - 26
            i_max_low_freq = i + 26
            i_min_low_freq_rad = i_min_low_freq*GRAD2RAD
            i_max_low_freq_rad = i_max_low_freq*GRAD2RAD
            p1 = np.round([size_half, size_half]).astype(int)
            p2 = np.round([size_half*(1-np.sin(i_min_low_freq_rad)), size_half*(1+np.cos(i_min_low_freq_rad))]).astype(int)
            p3 = np.round([size_half*(1-np.sin(i*GRAD2RAD)), size_half*(1+np.cos(i*GRAD2RAD))]).astype(int)
            p4 = np.round([size_half*(1-np.sin(i_max_low_freq_rad)), size_half*(1+np.cos(i_max_low_freq_rad))]).astype(int)
            pts = np.array([p1, p2, p3, p4], dtype=np.int32)
            bin_map = cv2.drawContours(np.zeros((size, size), dtype=np.uint8), [pts], -1, (255, 255, 255), -1)
            indices = np.ravel_multi_index(np.nonzero(bin_map), (size, size))
            if np.any(indices != low_last_indices):
                low_indices.append(np.array(sorted(indices, key=get_range), dtype=np.int32))
                low_ranges.append(range_map.flat[low_indices[-1]])
                low_indexer[i] = len(low_indices)-1

                low_last_indices = indices
            else:
                low_indexer[i] = low_indexer[i-1]

        high_indices = np.array(high_indices)
        high_ranges = np.array(high_ranges)
        low_indices = np.array(low_indices)
        low_ranges = np.array(low_ranges)
        self.occ_map_theta = self.theta[np.meshgrid(np.arange(factor/2, 1601-factor/2, factor, np.int),
                                                np.arange(factor/2, 1601-factor/2, factor, np.int))]
        # print('Calculated map successfully')
        np.savez('ogrid/OGrid_data/occ_map_{}.npz'.format(int(factor)), angle2cell_low=low_indices,
                 angle2cell_rad_low=low_ranges, angle2cell_high=high_indices, angle2cell_rad_high=high_ranges,
                 angles=self.occ_map_theta,
                 high_indexer=high_indexer, low_indexer=low_indexer)
        print('Map saved')
        self.angle2cell_low = low_indices
        self.angle2cell_rad_low = low_ranges
        self.angle2cell_high = high_indices
        self.angle2cell_rad_high = high_ranges
        self.low_indexer = low_indexer
        self.high_indexer = high_indexer

    def get_p(self):
        try:
            p = 1 - 1 / (1 + np.exp(self.grid))
        except RuntimeWarning:
            self.grid[np.nonzero(self.grid > 50)] = 50
            self.grid[np.nonzero(self.grid < -50)] = -50
            p = 1 - 1 / (1 + np.exp(self.grid))
            logger.debug('Overflow when calculating probability')
        return p

    def update_occ_zhou(self, msg, threshold):
        if msg.bearing < 0 or msg.bearing > 6399:
            # print(msg.bearing)
            return
        theta_rad = msg.bearing * np.pi / 3200.0
        obstacle_in_line = False
        self.contour_lock.acquire()
        if self.contours is not None and len(self.contours) > 0:
            contour, point = self.check_scan_line_intersection(theta_rad - np.pi)
            if contour is not None:
                theta_i, p1, p2 = self.calc_incident_angle(theta_rad - np.pi, contour, point)
                obstacle_in_line = True
        self.contour_lock.release()

        self.range_scale = msg.range_scale
        occ_grid = np.zeros((self.size, self.size), dtype=self.oLog_type)
        hit_ind = self.get_hit_inds(msg, threshold)
        if obstacle_in_line and not np.any(hit_ind):
            # r = grid2vehicle_rad(point[1], point[0], self.range_scale)
            # print(x, y, point)
            # hit_ind = np.array([np.round(r*msg.length/self.range_scale).astype(int)])
            # print(r, hit_ind[0])
            hit_factor = 0.5
            # hit_factor = 1
        else:
            hit_factor = 1
        logger.debug((hit_ind, hit_factor))
        if np.any(hit_ind):
            hit_extension_factor = GridSettings.hit_factor*msg.length/801
            data_ind_low = hit_ind - hit_extension_factor
            data_ind_high = hit_ind + hit_extension_factor

            if msg.chan2:
                cell_ind = False
                bearing_index = self.high_indexer[msg.bearing]
                for i in range(len(hit_ind)):
                    cell_ind = np.logical_or(np.logical_and(self.angle2cell_rad_high[bearing_index] > data_ind_low[i],
                                                            self.angle2cell_rad_high[bearing_index] < data_ind_high[i]),
                                             cell_ind)
                # cell_not_ind = np.nonzero(np.logical_not(cell_ind))
                # cell_ind = np.nonzero(cell_ind)
                cell_not_ind = np.logical_not(cell_ind)
                occ_grid.flat[self.angle2cell_high[bearing_index][cell_not_ind]] = self.p_log_free - self.p_log_zero
                if obstacle_in_line:
                    alpha = (wrapToPi(-self.occ_map_theta.flat[self.angle2cell_high[bearing_index][cell_ind]] - theta_rad)).clip(-0.05235987755982988, 0.05235987755982988)
                    tmp = GridSettings.kh_high * np.sin(alpha) / 2
                    p = (np.sin(tmp) / tmp) * (GridSettings.mu * np.sin(theta_i + alpha) ** 2)

                    p_max = np.max(p)
                    p_min = np.min(p)
                    p_max_min_2 = 2 * (p_max - p_min)
                    p = ((p - p_min) / p_max_min_2 + 0.5)*hit_factor
                    mask = 1 - p == 0
                    if np.any(mask):
                        log_p = np.log(p / (1 - p))
                        log_p[mask] = 5 * np.sign(p[mask] - 0.5)
                        occ_grid.flat[self.angle2cell_high[bearing_index][cell_ind]] = log_p
                    else:
                        occ_grid.flat[self.angle2cell_high[bearing_index][cell_ind]] = np.log(p / (1 - p))
                else:
                    occ_grid.flat[self.angle2cell_high[bearing_index][cell_ind]] = self.p_log_occ - self.p_log_zero
            else:
                # TODO: Use digitize instead of loop
                cell_ind = False
                bearing_index = self.low_indexer[msg.bearing]
                for i in range(len(hit_ind)):
                    cell_ind = np.logical_or(np.logical_and(self.angle2cell_rad_low[bearing_index] > data_ind_low[i],
                                                            self.angle2cell_rad_low[bearing_index] < data_ind_high[i]),
                                             cell_ind)
                # cell_not_ind = np.nonzero(np.logical_not(cell_ind))
                # cell_ind = np.nonzero(cell_ind)
                cell_not_ind = np.logical_not(cell_ind)
                occ_grid.flat[self.angle2cell_low[bearing_index][cell_not_ind]] = self.p_log_free - self.p_log_zero
                if np.any(cell_ind):
                    if obstacle_in_line:
                        alpha = (wrapToPi(-self.occ_map_theta.flat[self.angle2cell_low[bearing_index][cell_ind]]
                                          - theta_rad)).clip(-0.05235987755982988, 0.05235987755982988)
                        tmp = GridSettings.kh_low * np.sin(alpha) / 2
                        mask = tmp == 0
                        if np.any(mask):
                            try:
                                not_mask = np.logical_not(mask)
                                p = np.zeros(np.shape(mask))
                                p[not_mask] = (np.sin(tmp[not_mask]) / tmp[not_mask]) * (GridSettings.mu * np.sin(theta_i + alpha[not_mask]) ** 2)
                                p[mask] = GridSettings.mu * np.sin(theta_i + alpha[mask]) ** 2
                            except Exception as e:
                                a=1
                        else:
                            p = (np.sin(tmp) / tmp) * (GridSettings.mu * np.sin(theta_i + alpha) ** 2)

                        p_max = np.max(p)
                        p_min = np.min(p)
                        p_max_min_2 = 2 * (p_max - p_min)
                        if p_max_min_2 != 0:
                            p = ((p - p_min) / p_max_min_2 + 0.5)*hit_factor
                        mask = 1 - p == 0
                        if np.any(mask):
                            not_mask = np.logical_not(mask)
                            log_p = np.zeros(np.shape(mask))
                            log_p[not_mask] = np.log(p[not_mask] / (1 - p[not_mask]))
                            log_p[mask] = 5 * np.sign(p[mask] - 0.5)
                            occ_grid.flat[self.angle2cell_low[bearing_index][cell_ind]] = np.nan_to_num(log_p)
                        else:
                            occ_grid.flat[self.angle2cell_low[bearing_index][cell_ind]] = np.log(p / (1 - p))
                    else:
                        occ_grid.flat[self.angle2cell_low[bearing_index][cell_ind]] = self.p_log_occ - self.p_log_zero
        else:
            if msg.chan2:
                occ_grid.flat[self.angle2cell_high[self.high_indexer[msg.bearing]]] = self.p_log_free - self.p_log_zero
            else:
                occ_grid.flat[self.angle2cell_low[self.low_indexer[msg.bearing]]] = self.p_log_free - self.p_log_zero

        self.lock.acquire()
        self.occ2raw(occ_grid)
        self.grid = self.grid.clip(-10, 10)
        self.lock.release()

    def calc_incident_angle(self, angle, contour, point):
        """

        :param angle: scanline angle 0 deg straight up. in radians
        :param contour: id of intersection contour
        :param point: intersection point (x, y)
        :return:
        """
        contour = np.reshape(contour, (len(contour), 2))
        coord_sign = np.sign(contour - point)
        coord_sign_rolled = np.roll(coord_sign, 1, axis=0)
        i = np.argmax(np.all(coord_sign != coord_sign_rolled, axis=1))
        if i == 0 and not np.all(coord_sign[0, :] != coord_sign[-1, :]):
            eq0 = coord_sign[:, 0] == 0
            change_sign = coord_sign[:, 1] != coord_sign_rolled[:, 1]
            i = np.argmax(np.all(np.array([eq0.T, change_sign.T]).T, axis=1))
            if i == 0 and not coord_sign[0, 1] != coord_sign[-1, 1]:
                eq0 = coord_sign[:, 1] == 0
                change_sign = coord_sign[:, 0] != coord_sign_rolled[:, 0]
                i = np.argmax(np.all(np.array([change_sign.T, eq0.T]).T, axis=1))
                if i == 0 and not coord_sign[0, 0] != coord_sign[-1, 0]:
                    raise RuntimeWarning('Could not find intersecting contour')
        if coord_sign[i, 0] == 0 and coord_sign[i, 1] == 0:
            if i + 1 < np.shape(coord_sign)[0]:
                if i - 1 > -1:
                    c1 = contour[i - 1, :]
                    c2 = contour[i + 1, :]
                else:
                    c1 = contour[-1, :]
                    c2 = contour[i + 1, :]
            else:
                c1 = contour[0, :]
                c2 = contour[i - 1, :]
        else:
            if i + 1 < np.shape(coord_sign)[0]:
                c1 = contour[i, :]
                c2 = contour[i + 1, :]
            else:
                c1 = contour[i, :]
                c2 = contour[0, :]
        c_angle = np.arctan2((c2[0] - c1[0]), -(c2[1] - c1[1]))  # cv2 (x, y) => arctan(x / y)
        # print('contour_angle: {},\tline_angle: {},\tincident_angle: {}'.format(c_angle * 180.0 / np.pi, angle * 180.0 / np.pi, wrapToPiHalf(angle - c_angle) * 180.0 / np.pi))
        return wrapToPiHalf(angle - c_angle), c1, c2

    def check_scan_line_intersection(self, angle):
        '''
        intersection between scanline and contour
        :param angle: scanline angle(0 deg straight up)
        :return: intersecting contour, (intersecion point(x, y))
        '''
        if self.contours is None:
            return None, None
        angle_binary = np.zeros((self.i_max, self.j_max), dtype=np.uint8)
        # TODO: find outer coordinate
        cv2.line(angle_binary, (self.origin_i, self.origin_j),
                 (int(801*(1+np.sin(angle))), int(801*(1-np.cos(angle)))), (1, 1, 1), 1)  # cv2.line(im, (x, y), (x,y)
        if np.any(np.logical_and(self.bin_map, angle_binary)):
            for i in range(len(self.contours)):
                contour_binary = np.zeros((self.i_max, self.j_max), dtype=np.uint8)
                cv2.drawContours(contour_binary, self.contours, i, (1, 1, 1), 1)
                intersection_mat = np.logical_and(contour_binary, angle_binary)
                if np.any(intersection_mat):
                    points = np.array(np.nonzero(intersection_mat))
                    if angle < 0:
                        tmp = points[:, np.argmax(points[1, :])]
                        return self.contours[i], np.flip(tmp.T, 0)
                    elif angle > 0:
                        tmp = points[:, np.argmin(points[1, :])]
                        return self.contours[i], np.flip(tmp.T, 0)
                    else:
                        tmp = points[:, np.argmin(points[0, :])]
                        return self.contours[i], np.flip(tmp.T, 0)
        return None, None

    def interpolate_bins(self, msg):
        range_step = self.MAX_BINS / msg.dbytes
        new_data = np.zeros(self.MAX_BINS, dtype=np.uint8)
        updated = np.zeros(self.MAX_BINS, dtype=np.bool)
        try:
            for i in range(0, msg.dbytes):
                new_data[int(round(i * range_step))] = msg.data[i]
                updated[int(round(i * range_step))] = True
        except Exception as e:
            logger.debug('Mapping to unibins: {0}'.format(e))
        # new_data[np.nonzero(new_data < threshold)] = 0
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
        return new_data

    def calc_obstacles(self):
        self.grid = np.nan_to_num(self.grid)
        thresh = (self.grid > self.p_log_threshold).astype(dtype=np.uint8)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]

        # Removing small contours
        min_area = FeatureExtraction.min_area * self.range_scale
        new_contours = list()
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                new_contours.append(contour)
        im2 = cv2.drawContours(np.zeros(np.shape(self.grid), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)

        # dilating to join close contours and use safety margin
        k_size = np.round(CollisionSettings.obstacle_margin * 801.0 / self.range_scale).astype(int)
        im3 = cv2.dilate(im2, np.ones((k_size, k_size), dtype=np.uint8), iterations=1)
        # im3 = cv2.dilate(im2, self.kernel, iterations=FeatureExtraction.iterations)
        contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
        convex_contour_list = []
        for c in contours:
            convex_contour_list.append(cv2.convexHull(c, returnPoints=True))
        self.contour_lock.acquire()
        self.contours = convex_contour_list
        self.contour_lock.release()
        self.bin_map = np.zeros((self.i_max, self.j_max), dtype=np.uint8)

        cv2.drawContours(self.bin_map, self.contours, -1, (1, 1, 1), -1)
        # self.contour_as_line_list.clear()
        # for contour in contours:
        #     self.contour_as_line_list.append(cv2.fitLine(contour[0]))

        im = cv2.applyColorMap(((self.grid + 6) * 255.0 / 12.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        im = cv2.drawContours(im, self.contours, -1, (0, 0, 255), 5)
        # im[:, 800, :] = np.array([0, 0, 255])
        # im[800, :, :] = np.array([0, 0, 255])
        # im[801, :, :] = np.array([0, 0, 255])
        self.im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def get_obstacles(self):
        """
        calculates obstacles
        :return: image, contours
        """

        return self.im, self.contours

    # def get_hit_inds(self, msg, threshold):
    #     # hit indices by gradient threshold
    #     new_data = self.interpolate_bins(msg)
    #
    #     grad = np.gradient(new_data.astype(float))
    #     grad_max_ind = np.argmax(grad > threshold)
    #     try:
    #         if grad_max_ind != 0:
    #             threshold = new_data[grad_max_ind + 1]
    #         else:
    #             # logger.debug('No obstacles in this scanline, max grad: {}'.format(np.max(grad)))
    #             threshold = 256
    #     except IndexError:
    #         # logger.debug('No obstacles in this scanline, max grad: {}'.format(np.max(grad)))
    #         threshold = 256
    #
    #     return np.nonzero(new_data >= threshold)[0]

    def get_hit_inds(self, msg, threshold):
        threshold = min(max(threshold * msg.ad_span / 255.0 + msg.ad_low, 0), 255)
        # by normal threshold
        # new_data = self.interpolate_bins(msg)
        mean = np.mean(msg.data)
        max_val = np.max(msg.data)
        threshold = max(max_val - (max_val - mean) / 8, threshold)
        # print(threshold)
        return np.round(np.nonzero(msg.data >= threshold)[0] * self.MAX_BINS / msg.dbytes).astype(int)

    # def get_hit_inds(self, msg, threshold):
    #     threshold = min(max(threshold*msg.ad_span/255.0 + msg.ad_low, 0), 255)
    #     print(threshold)
    #     return np.round(np.nonzero(msg.data >= threshold)[0] * self.MAX_BINS / msg.dbytes).astype(int)


    # def get_hit_inds(self, msg, threshold):
    #     # Smooth graph
    #     # smooth = np.convolve(msg.data, np.full(GridSettings.smoothing_factor, 1.0/GridSettings.smoothing_factor),
    #     #                      mode='full')
    #     smooth = msg.data
    #     data_len = len(smooth)
    #     s1 = smooth[:-2]
    #     s2 = smooth[1:-1]
    #     s3 = smooth[2:]
    #
    #     # Find inital peaks and valleys
    #     peaks = (np.array(
    #         np.nonzero(np.logical_or(np.logical_and(s1 < s2, s2 > s3), np.logical_and(s1 < s2, s2 == s3)))).reshape(
    #         -1) + 1).tolist()
    #     valleys = (np.array(
    #         np.nonzero(np.logical_or(np.logical_and(s1 > s2, s2 < s3), np.logical_and(s1 > s2, s2 == s3)))).reshape(
    #         -1) + 1).tolist()
    #     if peaks[0] != 0 and peaks[0] < valleys[0]:
    #         valleys.insert(0, 0)
    #     if peaks[-1] != data_len - 1 and peaks[-1] > valleys[-1]:
    #         valleys.append(data_len - 1)
    #
    #     # Remove consecutive peaks or valleys
    #     signed_array = np.zeros(data_len, dtype=np.int8)
    #     signed_array[peaks] = 1
    #     signed_array[valleys] = -1
    #     sgn = signed_array[0]
    #     i_sgn = 0
    #     for i in range(1, data_len):
    #         if signed_array[i] == 1:
    #             if sgn == signed_array[i]:
    #                 peaks.remove(i_sgn)
    #             else:
    #                 sgn = 1
    #             i_sgn = i
    #         elif signed_array[i] == -1:
    #             if sgn == signed_array[i]:
    #                 valleys.remove(i_sgn)
    #             else:
    #                 sgn = -1
    #             i_sgn = i
    #
    #     # Remove peaks and valleys with primary factor lower than 5
    #     mask = np.logical_and(smooth[peaks] - smooth[valleys[:-1]] > 5, smooth[peaks] - smooth[valleys[1:]] > 5)
    #     peaks = (np.array(peaks)[mask]).tolist()
    #     signed_array = np.zeros(data_len, dtype=np.int8)
    #     signed_array[peaks] = 1
    #     signed_array[valleys] = -1
    #     sgn = signed_array[0]
    #     i_sgn = 0
    #     for i in range(1, data_len):
    #         if signed_array[i] == 1:
    #             if sgn == signed_array[i]:
    #                 peaks.remove(i_sgn)
    #             else:
    #                 sgn = 1
    #             i_sgn = i
    #         elif signed_array[i] == -1:
    #             if sgn == signed_array[i]:
    #                 if smooth[i] < smooth[i_sgn]:
    #                     valleys.remove(i_sgn)
    #                     i_sgn = i
    #                 else:
    #                     valleys.remove(i)
    #             else:
    #                 sgn = -1
    #                 i_sgn = i
    #
    #     # Return peaks with a primary factor higher than threshold and transform to 800 bin length
    #     smooth_peaks = smooth[peaks]
    #     smooth_valleys = smooth[valleys]
    #     mask = np.logical_or(smooth_peaks - smooth_valleys[:-1] > threshold,
    #                          smooth_peaks - smooth_valleys[1:] > threshold)
    #     return np.round(np.array(peaks)[mask] * self.MAX_BINS / msg.dbytes).astype(int)
        # print(np.array(peaks)[mask])
        # return smooth, peaks, valleys

    def randomize(self):
        self.range_scale = 30
        size = self.size // GridSettings.randomize_size
        rand = np.random.normal(0, GridSettings.randomize_max, (size, size))
        size2 = size//2
        size8 = size//6
        rand[size2-size8:size2+size8, size2-size8:size2+size8] = self.p_log_zero
        occ_grid = np.kron(rand,
                           np.ones((GridSettings.randomize_size, GridSettings.randomize_size)))

        nonzero = np.nonzero(occ_grid)
        if len(nonzero[0]) > 1:
            y1 = np.min(nonzero[0])
            y2 = np.max(nonzero[0]) + 1
            x1 = np.min(nonzero[1])
            x2 = np.max(nonzero[1]) + 1
            with self.lock:
                self.grid[y1 * self.cell_factor:y2 * self.cell_factor,
                x1 * self.cell_factor:x2 * self.cell_factor] = np.kron(
                    occ_grid[y1:y2, x1:x2], self.occ2raw_matrix)


if __name__=="__main__":
    import matplotlib.pyplot as plt

    # grid = OccupancyGrid(False, GridSettings.p_inital, GridSettings.p_occ, GridSettings.p_free, GridSettings.p_binary_threshold, 16)
    # grid.randomize()
    # grid2 = OccupancyGrid(False, GridSettings.p_inital, GridSettings.p_occ, GridSettings.p_free,
    #                      GridSettings.p_binary_threshold, 16)
    # grid3 = OccupancyGrid(False, GridSettings.p_inital, GridSettings.p_occ, GridSettings.p_free,
    #                      GridSettings.p_binary_threshold, 16)
    # grid.range_scale = 30
    # grid2.range_scale = 30
    # grid2.grid = grid.grid.copy()
    # grid3.range_scale = 30
    # grid3.grid = grid.grid.copy()
    # sgn = 10.0
    # for i in range(100):
    #     a = np.random.randint(-1, 1)
    #     b = np.random.randint(-1, 1)
    #     # a = -1
    #     # b = -1
    #     grid.trans3(a*sgn, b*sgn)
    #     grid2.trans4(a*sgn, b*sgn)
    #     # grid3.trans(sgn, sgn)
    #     # sgn = -sgn
    #
    # # for i in range(20):
    # #     grid.rotateImage(grid.grid, 5)
    #
    # plt.figure(1)
    # # grid.rot(5*np.pi/180)
    # plt.imshow(grid.grid)
    # plt.figure(2)
    # plt.imshow(grid2.grid)
    # plt.show()
    from messages.moosSonarMsg import *
    grid = OccupancyGrid(False, GridSettings.p_inital, GridSettings.p_occ, GridSettings.p_free,
                         GridSettings.p_binary_threshold, 16)
    data = np.load('ogrid/scanline.npz')['scanline']
    for s in data:
        plt.plot(s, color='b')
        msg = MoosSonarMsg()
        msg.data = s
        msg.dbytes = 300
        smooth, peaks, valleys = grid.get_hit_inds(msg, 120)
        plt.plot(smooth, color='g')
        plt.scatter(peaks, smooth[peaks], color='r', marker='o')
        plt.scatter(valleys, smooth[valleys], color='c', marker='o')
        plt.show()
        a=1

