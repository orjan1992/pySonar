from ogrid.rawGrid import RawGrid
import numpy as np
from settings import *
import logging
logger = logging.getLogger('OccupancyGrid')

class OccupancyGrid(RawGrid):
    def __init__(self, half_grid, p_zero, p_occ, p_free, p_bin_threshold, cell_factor):
        self.p_log_threshold = np.log(p_bin_threshold / (1 - p_bin_threshold))
        self.p_log_zero = np.log(p_zero / (1 - p_zero))
        self.p_log_occ = np.log(p_occ / (1 - p_occ))
        self.p_log_free = np.log(p_free / (1 - p_free))
        super().__init__(half_grid, self.p_log_zero)
        self.reliable = True
        self.cell_factor = cell_factor
        self.size = int((self.RES - 1) / cell_factor)
        try:
            with np.load('ogrid/OGrid_data/occ_map_{}_1601.npz'.format(int(cell_factor))) as data:
                self.new_map = data['new_map']
                self.angle2cell = data['angle2cell']
        except Exception as e:
            self.calc_map(cell_factor)

    def calc_map(self, factor):
        if factor % 2 != 0:
            raise ValueError('Wrong size reduction')
        size = int((self.RES - 1) / factor)
        new_map = np.zeros((self.N_ANGLE_STEPS, self.MAX_BINS, 2), dtype=np.uint32)
        angle2cell = [[] for x in range(self.N_ANGLE_STEPS)]
        for i in range(np.shape(self.map)[0]):
            print(i)
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

                    new_map[i, j, :] = cell_list[cell_ind]
                    if not cell_list[cell_ind] in angle2cell[i]:
                        angle2cell[i].append(cell_list[cell_ind])
        print('Calculated map successfully')
        np.savez('OGrid_data/occ_map_{}_1601.npz'.format(int(factor)), new_map=new_map, angle2cell=angle2cell)
        print('Map saved')

                #     new_map[i, j] = np.ravel_multi_index((cell_list[cell_ind][0], cell_list[cell_ind][1]), (size, size))
                # elif len(cell_list) > 0:
                #     new_map[i, j] = np.ravel_multi_index((cell_list[0][0], cell_list[0][1]), (size, size))
        # print('Convert to small map')
        # cell2grid_map = self.cell2grid_map
        # new_map = self.new_map
        # max_cells = 0
        # counter = 0
        # count_grid = np.zeros(np.shape(cell2grid_map))
        # for i in range(size):
        #     for j in range(size):
        #         count_grid[i, j] = len(cell2grid_map[i][j])
        #         if len(cell2grid_map[i][j]) > 64:
        #             counter += 1
        #         if len(cell2grid_map[i][j]) > max_cells:
        #             max_cells = len(cell2grid_map[i][j])
        # print('counter: {}'.format(counter))
        # reduced_map = np.zeros((self.N_ANGLE_STEPS, self.MAX_BINS, max_cells))
        # for i in range(self.N_ANGLE_STEPS):
        #     for j in range(self.MAX_BINS):
        #         for k in range(np.shape(cell2grid_map.flat[new_map[i, j]])):
        #             reduced_map[i, j, k] = cell2grid_map.flat[new_map[i, j]][k]
        # print('Saving results')
        # np.savez('OGrid_data/occ_map_new_{}_1601.npz'.format(int(factor)), bin2grid_map=reduced_map)

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
        # Try new threshold method
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

        # Update upper left corner of each occ grid cell
        # for i in range(self.MAX_BINS):
        #     if new_data[i] > threshold:
        #         self.update_cell(self.new_map[msg.bearing, i], self.p_log_occ + self.p_log_zero)
        #     else:
        #         self.update_cell(self.new_map[msg.bearing, i], self.p_log_free - self.p_log_zero)

        hit_ind = np.argmax(new_data > threshold)
        if hit_ind != 0:
            try:
                for i in range(hit_ind - 1):
                    self.update_cell(self.new_map[msg.bearing, i], self.p_log_free - self.p_log_zero)
                for i in range(hit_ind - 1, hit_ind + 2):
                    self.update_cell(self.new_map[msg.bearing, i], self.p_log_occ + self.p_log_zero)
            except IndexError:
                pass

        # Update the rest of the cells in one occ grid cell
        for cell in self.angle2cell[msg.bearing]:
            self.grid[cell[0]:cell[0] + self.cell_factor, cell[1]:cell[1] + self.cell_factor] = self.grid[cell[0], cell[1]]


        # bearing_diff = msg.bearing - self.last_bearing

        # for i in range(self.MAX_BINS):


        # beam_half = 27
        # if msg.chan2:
        #     beam_half = 13
        #
        # if math.fabs(bearing_diff) <= msg.step:
        #     if bearing_diff > 0:
        #         value_gain = (new_data.astype(float) - self.last_data) / bearing_diff
        #         for n in range(self.last_bearing, msg.bearing + 1):
        #             for i in range(0, self.MAX_CELLS):
        #                 self.grid.flat[RawGrid.map[n, :, i]] += new_data + (n - self.last_bearing) * value_gain
        #         for n in range(msg.bearing + 1, msg.bearing + beam_half):
        #             for i in range(0, self.MAX_CELLS):
        #                 self.grid.flat[RawGrid.map[n, :, i]] += new_data
        #     else:
        #         value_gain = (new_data.astype(float) - self.last_data) / (-bearing_diff)
        #         for n in range(msg.bearing, self.last_bearing + 1):
        #             for i in range(0, self.MAX_CELLS):
        #                 self.grid.flat[RawGrid.map[n, :, i]] += new_data + (n - msg.bearing) * value_gain
        #         for n in range(msg.bearing - beam_half, msg.bearing):
        #             for i in range(0, self.MAX_CELLS):
        #                 self.grid.flat[RawGrid.map[n, :, i]] += new_data
        # else:
        #     for n in range(msg.bearing - beam_half, msg.bearing + beam_half):
        #         for i in range(0, self.MAX_CELLS):
        #             self.grid.flat[RawGrid.map[n, :, i]] += new_data
        #
        # self.last_bearing = msg.bearing
        # self.last_data = new_data

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
        im3 = cv2.dilate(im2, FeatureExtraction.kernel, iterations=FeatureExtraction.iterations)
        contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
        im = cv2.applyColorMap(((self.grid + 6)*255.0 / 12.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        im = cv2.drawContours(im, contours, -1, (255, 0, 0), 2)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), contours

if __name__=="__main__":
    grid = OccupancyGrid(True, 0.7, 16)
    # grid.calc_map(2)
    # grid.calc_map(4)
    # grid.calc_map(8)
    # grid.calc_map(16)
    # grid.calc_map(32)
    # print('Finished sucsessfully')
    # with np.load('OGrid_data/occ_map_2_1601.npz') as data:
    #     new_map = data['new_map']
    #     cell2grid_map = data['cell2grid_map']
    a = 1