from ogrid.rawGrid import RawGrid
import numpy as np
from settings import *
from coordinate_transformations import wrapTo2Pi, wrapToPi, wrapToPiHalf
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
                # for i in range(self.N_ANGLE_STEPS):
                #     print(wrapToPi(np.mean(-self.occ_map_theta.flat[self.angle2cell_high[i]]) - i*np.pi/3200.0))
        except Exception as e:
            self.calc_occ_map(cell_factor)

    def occ2raw(self, occ_grid):
        nonzero = np.nonzero(occ_grid)
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
        # TODO: Should have all cells in cone. sort by bin and left to right
        if factor % 2 != 0:
            raise ValueError('Wrong size reduction')
        size = int((self.RES - 1) / factor)
        f2 = factor // 2
        angle2cell = [[] for x in range(self.N_ANGLE_STEPS)]
        angle2cell_rad = [[] for x in range(self.N_ANGLE_STEPS)]
        for i in range(self.N_ANGLE_STEPS):
            for j in range(np.shape(self.map)[1]):
                cell_list = []
                for cell in self.map[i, j][self.map[i, j] != 0]:
                    row, col = np.unravel_index(cell, (self.RES, self.RES))
                    new_row = int(row // factor)
                    new_col = int(col // factor)

                    cell_list.append((new_row, new_col))
                if len(cell_list) > 0:
                    cell_ind = 0
                    if len(cell_list) > 1:
                        r = self.r_unit.flat[self.map[i, j][self.map[i, j] != 0]]
                        theta_grad = self.theta_grad.flat[self.map[i, j][self.map[i, j] != 0]]
                        cell_ind = np.argmin(np.abs(j - r) * 6400 + np.abs(i - theta_grad))
                    if not cell_list[cell_ind] in angle2cell[i]:
                        angle2cell[i].append(cell_list[cell_ind])
            #             angle2cell_rad[i].append(((801 - cell_list[cell_ind][0] + f2) ** 2 + (
            #                         cell_list[cell_ind][1] - 801 + f2) ** 2) ** 0.5)
            # angle2cell[i] = [x for _, x in sorted(zip(angle2cell_rad[i], angle2cell[i]))]
            # angle2cell_rad[i].sort()
            angle2cell[i] = np.ravel_multi_index(np.array(angle2cell[i]).T, (size, size))
            print(i)

        range_map = self.r_unit[np.meshgrid(np.arange(factor/2, 1601-factor/2, factor, np.int), np.arange(factor/2, 1601-factor/2, factor, np.int))]*801

        def get_range(ind):
            return range_map.flat[ind]

        high_indices = [[] for x in range(self.N_ANGLE_STEPS)]
        low_indices = [[] for x in range(self.N_ANGLE_STEPS)]
        high_ranges = [[] for x in range(self.N_ANGLE_STEPS)]
        low_ranges = [[] for x in range(self.N_ANGLE_STEPS)]
        for i in range(self.N_ANGLE_STEPS):
            print('\t' + str(i))
            i_min_high_freq = i - 13
            i_max_high_freq = i + 13
            high_freq = angle2cell[i_min_high_freq]
            for j in range(i_min_high_freq, i_max_high_freq):
                if j+1 < self.N_ANGLE_STEPS:
                    high_freq = np.union1d(high_freq, angle2cell[j+1])
                else:
                    high_freq = np.union1d(high_freq, angle2cell[j + 1 - self.N_ANGLE_STEPS])
            high_indices[i] = sorted(high_freq, key=get_range)
            high_ranges[i] = range_map.flat[high_indices[i]]

            i_min_low_freq = i - 26
            i_max_low_freq = i + 26
            low_freq = angle2cell[i_min_low_freq]
            for j in range(i_min_low_freq, i_max_low_freq):
                if j+1 < self.N_ANGLE_STEPS:
                    low_freq = np.union1d(low_freq, angle2cell[j+1])
                else:
                    low_freq = np.union1d(low_freq, angle2cell[j + 1 - self.N_ANGLE_STEPS])
            low_indices[i] = sorted(low_freq, key=get_range)
            low_ranges[i] = range_map.flat[low_indices[i]]
        print('Calculated map successfully')
        np.savez('ogrid/OGrid_data/occ_map_{}.npz'.format(int(factor)), angle2cell_low=low_indices,
                 angle2cell_rad_low=low_ranges, angle2cell_high=high_indices, angle2cell_rad_high=high_ranges,
                 angles=self.theta[np.meshgrid(np.arange(factor/2, 1601-factor/2, factor, np.int),
                                                np.arange(factor/2, 1601-factor/2, factor, np.int))])
        print('Map saved')
        self.angle2cell_low = low_indices
        self.angle2cell_rad_low = low_ranges
        self.angle2cell_high = high_indices
        self.angle2cell_rad_high = high_ranges

    def get_p(self):
        try:
            p = 1 - 1 / (1 + np.exp(self.grid.clip(-6.0, 6.0)))
        except RuntimeWarning:
            self.grid[np.nonzero(self.grid > 50)] = 50
            self.grid[np.nonzero(self.grid < -50)] = -50
            p = 1 - 1 / (1 + np.exp(self.grid))
            logger.debug('Overflow when calculating probability')
        return p

    def update_occ_zhou(self, msg, threshold):
        try:
            occ_grid = np.zeros((self.size, self.size), dtype=self.oLog_type)
            new_data = self.interpolate_bins(msg)
            grad = np.gradient(new_data.astype(float))
            grad_max_ind = np.argmax(grad > threshold)
            try:
                if grad_max_ind != 0:
                    threshold = new_data[grad_max_ind + 1]
                else:
                    # logger.debug('No obstacles in this scanline, max grad: {}'.format(np.max(grad)))
                    threshold = 256
            except IndexError:
                # logger.debug('No obstacles in this scanline, max grad: {}'.format(np.max(grad)))
                threshold = 256
            hit_ind = np.argmax(new_data >= threshold)

            theta_rad = msg.bearing * np.pi / 3200.0


            # print('hit_ind: {},\tthresh: {}'.format(hit_ind, threshold))
            if hit_ind == 0:
                if msg.chan2:
                    for cell in self.angle2cell_high[msg.bearing]:
                        occ_grid.flat[cell] = self.p_log_free - self.p_log_zero
                else:
                    for cell in self.angle2cell_low[msg.bearing]:
                        occ_grid.flat[cell] = self.p_log_free - self.p_log_zero
            else:
                if self.contours is not None:
                    contour, point = self.check_scan_line_intersection(theta_rad - np.pi)
                    if contour is not None:
                        theta_i, p1, p2 = self.calc_incident_angle(theta_rad - np.pi, contour, point)
                    else:
                        theta_i = np.pi / 2
                else:
                    theta_i = np.pi / 2

                hit_ind_low = hit_ind - GridSettings.hit_factor
                hit_ind_high = hit_ind + GridSettings.hit_factor
                if msg.chan2:
                    i_max = len(self.angle2cell_high[msg.bearing])
                else:
                    i_max = len(self.angle2cell_low[msg.bearing])

                if msg.chan2:
                    ind0 = np.argmax(self.angle2cell_rad_high[msg.bearing] > hit_ind_low)
                    ind1 = np.argmax(self.angle2cell_rad_high[msg.bearing] >= hit_ind_high)
                    for i in range(ind0):
                        occ_grid.flat[self.angle2cell_high[msg.bearing][i]] = self.p_log_free - self.p_log_zero
                    if ind1 == 0:
                        ind1 = len(self.angle2cell_rad_high[msg.bearing])
                    P = np.zeros(ind1-ind0)

                    for i in range(ind0, ind1):
                        alpha = wrapToPi(-self.occ_map_theta.flat[self.angle2cell_high[msg.bearing][i]] - theta_rad)
                        tmp = GridSettings.kh_high * np.sin(alpha) / 2
                        P[i - ind0] = (np.sin(tmp) / tmp)*(GridSettings.mu * np.sin(theta_i+alpha)**2)
                    P_max = np.max(P)
                    P_min = np.min(P)
                    P_max_min_2 = 2*(P_max-P_min)
                    for i in range(ind0, ind1):
                        p = (P[i - ind0] - P_min)/P_max_min_2 + 0.5
                        if p < 0 or p > 1:
                            logger.error('Probability is invalid. p={}'.format(p))
                        occ_grid.flat[self.angle2cell_high[msg.bearing][i]] = np.log(p / (1-p))

                else:
                    ind0 = np.argmax(self.angle2cell_rad_low[msg.bearing] > hit_ind_low)
                    ind1 = np.argmax(self.angle2cell_rad_low[msg.bearing] >= hit_ind_low)
                    for i in range(ind0):
                        occ_grid.flat[self.angle2cell_low[msg.bearing][i]] = self.p_log_free - self.p_log_zero
                    P = np.zeros(ind1-ind0)

                    for i in range(ind0, ind1):
                        alpha = wrapToPi(-self.occ_map_theta.flat[self.angle2cell_low[msg.bearing][i]] - theta_rad)
                        tmp = GridSettings.kh_low * np.sin(alpha) / 2
                        P[i - ind0] = (np.sin(tmp) / tmp)*(GridSettings.mu * np.sin(theta_i+alpha)**2)
                    P_max = np.max(P)
                    P_min = np.min(P)
                    P_max_min_2 = 2*(P_max-P_min)
                    for i in range(ind0, ind1):
                        p = (P[i - ind0] - P_min)/P_max_min_2 + 0.5
                        if p < 0 or p > 1:
                            logger.error('Probability is invalid. p={}'.format(p))
                        occ_grid.flat[self.angle2cell_low[msg.bearing][i]] = np.log(p / (1-p))

            self.lock.acquire()
            self.occ2raw(occ_grid)
            self.lock.release()
        except Exception as e:
            import traceback
            traceback.print_exc()

    def calc_incident_angle(self, angle, contour, point):
        """

        :param angle: scanline angle 0 deg straight up. in radians
        :param contour: id of intersection contour
        :param point: intersection point (x, y)
        :return:
        """
        # TODO: Fix angle to 0 deg at north
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

    def get_obstacles(self):

        thresh = (self.grid > self.p_log_threshold).astype(dtype=np.uint8)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]

        # Removing small contours
        # TODO: Should min_area be dependent on range?
        new_contours = list()
        for contour in contours:
            if cv2.contourArea(contour) > FeatureExtraction.min_area:
                new_contours.append(contour)
        im2 = cv2.drawContours(np.zeros(np.shape(self.grid), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)

        # dilating to join close contours
        # TODO: maybe introduce safety margin in this dilation
        im3 = cv2.dilate(im2, self.kernel, iterations=FeatureExtraction.iterations)
        self.contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
        self.bin_map = np.zeros((self.i_max, self.j_max), dtype=np.uint8)

        cv2.drawContours(self.bin_map, self.contours, -1, (1, 1, 1), -1)
        # self.contour_as_line_list.clear()
        # for contour in contours:
        #     self.contour_as_line_list.append(cv2.fitLine(contour[0]))

        im = cv2.applyColorMap(((self.grid + 6)*255.0 / 12.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        im = cv2.drawContours(im, self.contours, -1, (255, 0, 0), 2)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), self.contours

if __name__=="__main__":
    import time
    import matplotlib.pyplot as plt
    from messages.moosSonarMsg import MoosSonarMsg
    grid = OccupancyGrid(False, 0.3, 0.9, 0.7, 0.75, 16)

    def test():
        occ = np.zeros((grid.size, grid.size), grid.oLog_type)
        grid.occ2raw(occ)

    for i in range(100):
        test()
        print(i)

    # a = np.load('collision_avoidance/test.npz')['olog']
    # grid.grid[:np.shape(a)[0], :np.shape(a)[1]] = a/8.0 -2
    # im, countours = grid.get_obstacles()
    #
    # l_list = []
    # # map = np.zeros((grid.i_max, grid.j_max), dtype=np.uint8)
    # # for c in countours:
    # #     line = cv2.fitLine(c[0], cv2.DIST_L2, 0, 1, 0.1)
    # #     cv2.line(map, (line[0], line[3]), (line[1], line[3]), (255, 255, 255), 1)
    # angle = 2600.0*np.pi / 3200 - np.pi
    # id, point = grid.check_scan_line_intersection(angle)
    # cv2.line(im, (801, 801), (int(801*(1+np.sin(angle))), int(801*(1-np.cos(angle)))), (0, 0, 255), 5)
    # try:
    #     int_angle, p1, p2 = grid.calc_incident_angle(angle, id, point)
    #
    #     # cv2.circle(im, (point[0], point[1]), 10, (0, 255, 0), 3)
    #     # for contour in countours:
    #     #     # cv2.drawContours(im, [np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))], 0, (0, 0, 255), 1)
    #     #     for p in contour:
    #     #         cv2.circle(im, (p[0][0], p[0][1]), 5, (255, 0, 0), 5)
    #     # cv2.drawContours(im, countours, id, (0, 0, 255), 5)
    #     cv2.circle(im, (point[0], point[1]), 5, (255, 0, 0), 2)
    #     cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 5)
    # except:
    #     pass
    # # cv2.imshow('sdf', im)
    # # cv2.waitKey()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()
    # a = 1