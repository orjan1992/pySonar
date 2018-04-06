from ogrid.rawGrid import RawGrid
import numpy as np
from settings import *
from coordinate_transformations import wrap2twopi
import logging
logger = logging.getLogger('OccupancyGrid')

class OccupancyGrid(RawGrid):
    # counter = None
    # sign = None
    occ_map = np.zeros((RawGrid.N_ANGLE_STEPS), dtype=object)
    contour_as_line_list = []
    bin_map = np.zeros((RawGrid.i_max, RawGrid.j_max), dtype=np.uint8)
    # TODO: local occ grid with shape=(RES/cellFactor, RES/cellFactor) update to res with np.kron(occ, np.ones(factor, factor)

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
        self.occ_map_range = np.zeros((self.size, self.size))
        self.occ_map_theta = np.zeros((self.size, self.size))  # TODO: Should be in 1/16 grad
        self.occ2raw_matrix = np.ones((cell_factor, cell_factor))
        try:
            with np.load('ogrid/OGrid_data/occ_map_{}_1601.npz'.format(int(cell_factor))) as data:
                self.angle2cell = data['angle2cell']
                self.angle2cell_rad = data['angle2cell_rad']
        except Exception as e:
            self.calc_map(cell_factor)

    def occ2raw(self, occ_grid):
        self.grid += np.kron(occ_grid, self.occ2raw_matrix)

    def raw2occ(self):
        occ_grid = np.ones((self.size, self.size), dtype=self.oLog_type)
        for i in range(self.size):
            for j in range(self.size):
                occ_grid[i, j] = np.mean(self.grid[i:i+self.cell_factor, j:j+self.cell_factor])
        return occ_grid

    def calc_map(self, factor):
        if factor % 2 != 0:
            raise ValueError('Wrong size reduction')
        f2 = factor // 2
        angle2cell = [[] for x in range(self.N_ANGLE_STEPS)]
        angle2cell_rad = [[] for x in range(self.N_ANGLE_STEPS)]
        for i in range(np.shape(self.map)[0]):
            for j in range(np.shape(self.map)[1]):
                cell_list = []
                for cell in self.map[i, j][self.map[i, j] != 0]:
                    row, col = np.unravel_index(cell, (self.RES, self.RES))
                    new_row = (row // factor) * factor
                    new_col = (col // factor) * factor

                    cell_list.append((new_row, new_col))
                if len(cell_list) > 0:
                    cell_ind = 0
                    if len(cell_list) > 1:
                        r = self.r_unit.flat[self.map[i, j][self.map[i, j] != 0]]
                        theta_grad = self.theta_grad.flat[self.map[i, j][self.map[i, j] != 0]]
                        cell_ind = np.argmin(np.abs(j-r)*6400 + np.abs(i - theta_grad))
                    if not cell_list[cell_ind] in angle2cell[i]:
                        angle2cell[i].append(cell_list[cell_ind])
                        angle2cell_rad[i].append(((801 - cell_list[cell_ind][0] + f2)**2 + (cell_list[cell_ind][1] - 801 + f2)**2)**0.5)
            angle2cell[i] = [x for _, x in sorted(zip(angle2cell_rad[i], angle2cell[i]))]
            angle2cell_rad[i].sort()
            print(i)
        print('Calculated map successfully')
        np.savez('ogrid/OGrid_data/occ_map_{}_1601.npz'.format(int(factor)), angle2cell=angle2cell, angle2cell_rad=angle2cell_rad)
        print('Map saved')
        self.angle2cell_rad = angle2cell_rad
        self.angle2cell = angle2cell

    # def calc_cell_rad(self, factor):
    #     self.rad_map = [[] for i in range(self.N_ANGLE_STEPS)]
    #     f2 = factor // 2
    #     for i in range(len(self.angle2cell)):
    #         for j in range(len(self.angle2cell[i])):
    #             self.rad_map[i].append(((801 - self.angle2cell[i][j][0] + f2)**2 + (self.angle2cell[i][j][1] - 801 + f2)**2)**0.5)
    #     np.savez('ogrid/OGrid_data/occ_map_rad_{}_1601.npz'.format(int(factor)), angle2cell_rad=self.rad_map)

    def calc_occ_map(self, factor):
        # TODO: Should have all cells in cone. sort by bin and left to right
        if factor % 2 != 0:
            raise ValueError('Wrong size reduction')
        size = int((self.RES - 1) / factor)
        f2 = factor // 2
        angle2cell = [[] for x in range(self.N_ANGLE_STEPS)]
        angle2cell_rad = [[] for x in range(self.N_ANGLE_STEPS)]
        for i in range(np.shape(self.map)[0]):
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
                        angle2cell_rad[i].append(((801 - cell_list[cell_ind][0] + f2) ** 2 + (
                                    cell_list[cell_ind][1] - 801 + f2) ** 2) ** 0.5)
            angle2cell[i] = [x for _, x in sorted(zip(angle2cell_rad[i], angle2cell[i]))]
            angle2cell_rad[i].sort()
            print(i)
        print('Calculated map successfully')
        np.savez('ogrid/OGrid_data/occ_map_{}_1601.npz'.format(int(factor)), angle2cell=angle2cell,
                 angle2cell_rad=angle2cell_rad)
        print('Map saved')
        self.angle2cell_rad = angle2cell_rad
        self.angle2cell = angle2cell


    def update_cell(self, indices, value):
        # self.grid[indices[0]:(indices[0] + self.cell_factor), indices[1]:(indices[1] + self.cell_factor)] += value
        self.grid[indices[0], indices[1]] += value

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
        occ_grid = np.zeros((self.size, self.size), dtype=self.oLog_type)
        new_data = self.interpolate_bins(msg)
        hit_ind = np.argmax(new_data > threshold)

        # TODO: Calculate incident angle
        theta_grad = msg.bearing*3200.0/np.pi
        sonar_line = ((self.origin_i, self.origin_j), (int(801 - 1132.78*np.sin(theta_grad)), int(1132.78*np.cos(theta_grad) - 801)))
        for line in self.contour_as_line_list:
            # TODO: cv2.intersect
            # cv2.intersectConvexConvex()
            theta_i = 0.5

        # TODO: Def k and h and mu
        k = h = mu = 1

        if hit_ind == 0:
            for cell in self.occ_map[msg.bearing]:
                occ_grid.flat[cell] = self.p_log_free - self.p_log_zero
        else:
            hit_ind -= GridSettings.hit_factor
            i_max = len(self.occ_map[msg.bearing])
            i = 0
            exit_loop = False
            # alpha = 1.5*np.pi/180
            # if msg.chan2:
            #     alpha *= 0.5
            # p_d_a = np.sin(k * h * np.sin(alpha) / 2)
            # p_max =
            while i < i_max:
                for j in range(len(self.occ_map[msg.bearing][i])):
                    # TODO: Maybe extend range by a factor
                    if self.occ_map_range.flat[self.occ_map[msg.bearing][i][j]] < hit_ind:
                        occ_grid.flat[self.occ_map[msg.bearing][i][j]] = self.p_log_free - self.p_log_zero
                    else:
                        alpha = np.abs(self.occ_map_theta.flat[self.occ_map[msg.bearing][i][j]] - msg.bearing)
                        P_DI = np.sin(k * h * np.sin(alpha) / 2)
                        P_TS = mu * np.sin(theta_i)**2
                        occ_grid.flat[self.occ_map[msg.bearing][i][j]] = self.p_log_occ - self.p_log_zero  # P_DI*P_TS
                        # TODO: calculate prob
                        if not exit_loop:
                            i_max = np.min(i + GridSettings.hit_factor, i_max)
                            exit_loop = True
                i += 1
        self.occ2raw(occ_grid)

    def calc_incident_angle(self, angle, contour_id, point):
        # TODO: Fix angle to 0 deg at north
        contour = self.contours[contour_id]
        contour = np.reshape(contour, (len(contour), 2))
        coord_sign = np.sign(contour - point)
        i = np.argmax(np.all(coord_sign != np.roll(coord_sign, 1, axis=0), axis=1))
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
        c_angle = np.arctan2((c2[1] - c1[1]), -(c2[0] - c1[0]))
        print(c_angle*180.0/np.pi)
        return wrap2twopi(angle - np.pi - c_angle), c1, c2

    def check_scan_line_intersection(self, angle):
        angle_binary = np.zeros((self.i_max, self.j_max), dtype=np.uint8)
        # TODO: find outer coordinate
        cv2.line(angle_binary, (self.origin_i, self.origin_j),
                 (int(801 - 801 * np.sin(angle)), int(801 * np.cos(angle) + 801)), (1, 1, 1), 1)
        if np.any(np.logical_and(self.bin_map, angle_binary)):
            for i in range(len(self.contours)):
                contour_binary = np.zeros((self.i_max, self.j_max), dtype=np.uint8)
                cv2.drawContours(contour_binary, self.contours, i, (1, 1, 1), 1)
                intersection_mat = np.logical_and(contour_binary, angle_binary)
                if np.any(intersection_mat):
                    points = np.array(np.nonzero(intersection_mat))
                    if angle < 0:
                        tmp = points[:, np.argmax(points[1, :])]
                        return i, np.array([tmp[1], tmp[0]])
                    elif angle > 0:
                        tmp = points[:, np.argmin(points[1, :])]
                        return i, np.array([tmp[1], tmp[0]])
                    else:
                        tmp = points[:, np.argmin(points[0, :])]
                        return i, np.array([tmp[1], tmp[0]])
        return -1, None

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

    def auto_update_zhou(self, msg, threshold):
        # Try new threshold method
        new_data = self.interpolate_bins(msg)
        grad = np.gradient(new_data.astype(float))
        grad_max_ind = np.argmax(grad > threshold)
        try:
            if grad_max_ind != 0:
                threshold = new_data[grad_max_ind+1]
            else:
                # logger.debug('No obstacles in this scanline, max grad: {}'.format(np.max(grad)))
                return
        except IndexError:
            # logger.debug('No obstacles in this scanline, max grad: {}'.format(np.max(grad)))
            return

        # if self.counter is None:
        #     self.counter = 1
        #     self.sign = 1
        # if self.counter > 6398 or self.counter < 1:
        #     self.sign = -self.sign
        # self.counter += self.sign
        # msg.bearing = self.counter
        # new_data = np.zeros(800, dtype=np.uint8)
        # new_data[700:720] = 255
        # threshold = 200
        self.lock.acquire()
        hit_ind = np.argmax(new_data > threshold)
        if hit_ind == 0:
            for cell in self.angle2cell[msg.bearing]:
                self.update_cell(cell, self.p_log_free - self.p_log_zero)
        else:
            for i in range(len(self.angle2cell_rad[msg.bearing])):
                if self.angle2cell_rad[msg.bearing][i] < hit_ind:
                    self.update_cell(self.angle2cell[msg.bearing][i], self.p_log_free - self.p_log_zero)
                else:
                    self.update_cell(self.angle2cell[msg.bearing][i], self.p_log_occ - self.p_log_zero)
                    break
        # Update the rest of the cells in one occ grid cell
        for cell in self.angle2cell[msg.bearing]:
            self.grid[cell[0]:cell[0] + self.cell_factor, cell[1]:cell[1] + self.cell_factor] = self.grid[cell[0], cell[1]]
        self.lock.release()
        # beam_half = 27
        # if msg.chan2:
        #     beam_half = 13
        # bearing_low = msg.bearing - beam_half
        # bearing_high = msg.bearing + beam_half
        # hit_ind = np.argmax(new_data > threshold)
        # if hit_ind != 0:
        #     try:
        #         for i in range(hit_ind - 1):
        #             self.update_cell(self.new_map[msg.bearing, i], self.p_log_free - self.p_log_zero)
        #             self.update_cell(self.new_map[bearing_low, i], self.p_log_free - self.p_log_zero)
        #             self.update_cell(self.new_map[bearing_high, i], self.p_log_free - self.p_log_zero)
        #         for i in range(hit_ind - 1, hit_ind + 2):
        #             self.update_cell(self.new_map[msg.bearing, i], self.p_log_occ + self.p_log_zero)
        #             self.update_cell(self.new_map[bearing_low, i], self.p_log_occ + self.p_log_zero)
        #             self.update_cell(self.new_map[bearing_high, i], self.p_log_occ + self.p_log_zero)
        #     except IndexError:
        #         pass
        # # else:
        # #     for i in range(self.MAX_BINS):
        # #         self.update_cell(self.new_map[msg.bearing, i], self.p_log_free - self.p_log_zero)
        # #         self.update_cell(self.new_map[bearing_low, i], self.p_log_free - self.p_log_zero)
        # #         self.update_cell(self.new_map[bearing_high, i], self.p_log_free - self.p_log_zero)
        #
        # # Update the rest of the cells in one occ grid cell
        # for cell in self.angle2cell[msg.bearing]:
        #     self.grid[cell[0]:cell[0] + self.cell_factor, cell[1]:cell[1] + self.cell_factor] = self.grid[cell[0], cell[1]]
        # for cell in self.angle2cell[bearing_low]:
        #     self.grid[cell[0]:cell[0] + self.cell_factor, cell[1]:cell[1] + self.cell_factor] = self.grid[cell[0], cell[1]]
        # for cell in self.angle2cell[bearing_high]:
        #     self.grid[cell[0]:cell[0] + self.cell_factor, cell[1]:cell[1] + self.cell_factor] = self.grid[cell[0], cell[1]]

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
    grid = OccupancyGrid(True, 0.3, 0.9, 0.7, 0.75, 16)
    a = np.load('collision_avoidance/test.npz')['olog']
    grid.grid[:np.shape(a)[0], :np.shape(a)[1]] = a/8.0 -2
    im, countours = grid.get_obstacles()

    l_list = []
    # map = np.zeros((grid.i_max, grid.j_max), dtype=np.uint8)
    # for c in countours:
    #     line = cv2.fitLine(c[0], cv2.DIST_L2, 0, 1, 0.1)
    #     cv2.line(map, (line[0], line[3]), (line[1], line[3]), (255, 255, 255), 1)
    angle = 4502.0*np.pi / 3200
    id, point = grid.check_scan_line_intersection(angle)
    cv2.line(im, (801, 801), (int(801 - 801*np.sin(angle)), int(801*np.cos(angle) + 801)), (0, 0, 255), 5)
    int_angle, p1, p2 = grid.calc_incident_angle(angle, id, point)

    # cv2.circle(im, (point[0], point[1]), 10, (0, 255, 0), 3)
    # for contour in countours:
    #     # cv2.drawContours(im, [np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))], 0, (0, 0, 255), 1)
    #     for p in contour:
    #         cv2.circle(im, (p[0][0], p[0][1]), 5, (255, 0, 0), 5)
    # cv2.drawContours(im, countours, id, (0, 0, 255), 5)
    cv2.circle(im, (point[0], point[1]), 5, (255, 0, 0), 2)
    cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 5)
    # cv2.imshow('sdf', im)
    # cv2.waitKey()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()
    print(int_angle*180/np.pi)