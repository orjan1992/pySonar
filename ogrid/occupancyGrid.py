from ogrid.rawGrid import RawGrid
import numpy as np
import math
import logging
logger = logging.getLogger('OccupancyGrid')

class OccupancyGrid(RawGrid):
    def __init__(self, half_grid, p_m, cellfactor):
        super().__init__(half_grid, p_m)
        self.cell_size = cellfactor
        try:
            with np.load('OGrid_data/occ_map_{}_1601.npz'.format(int(cellfactor))) as data:
                self.new_map = data['new_map']
                self.cell2grid_map = data['cell2grid_map']
        except:
            self.calc_map(cellfactor)

    def calc_map(self, factor):
        if factor % 2 != 0:
            raise ValueError('Wrong size reduction')
        size = int((self.RES - 1) / factor)
        # new_map = np.zeros((self.N_ANGLE_STEPS, self.MAX_BINS), dtype=np.uint32)
        # cell2grid_map = [[[] for x in range(size)] for y in range(size)]
        # for i in range(np.shape(self.map)[0]):
        #     print(i)
        #     for j in range(np.shape(self.map)[1]):
        #         cell_list = []
        #         for cell in self.map[i, j][self.map[i, j] != 0]:
        #             row, col = np.unravel_index(cell, (self.RES, self.RES))
        #             new_row = row // factor
        #             new_col = col // factor
        #             cell2grid_map[new_row][new_col].append(cell)
        #             cell_list.append((new_row, new_col))
        #         if len(cell_list) > 1:
        #             r = self.r_unit.flat[self.map[i, j][self.map[i, j] != 0]]
        #             theta_grad = self.theta_grad.flat[self.map[i, j][self.map[i, j] != 0]]
        #             cell_ind = np.argmin(np.abs(j-r)*6400 + np.abs(i - theta_grad))
        #             new_map[i, j] = np.ravel_multi_index((cell_list[cell_ind][0], cell_list[cell_ind][1]), (size, size))
        #         elif len(cell_list) > 0:
        #             new_map[i, j] = np.ravel_multi_index((cell_list[0][0], cell_list[0][1]), (size, size))
        # print('Convert to small map')
        cell2grid_map = self.cell2grid_map
        new_map = self.new_map
        max_cells = 0
        counter = 0
        count_grid = np.zeros(np.shape(cell2grid_map))
        for i in range(size):
            for j in range(size):
                count_grid[i, j] = len(cell2grid_map[i][j])
                if len(cell2grid_map[i][j]) > 64:
                    counter += 1
                if len(cell2grid_map[i][j]) > max_cells:
                    max_cells = len(cell2grid_map[i][j])
        print('counter: {}'.format(counter))
        reduced_map = np.zeros((self.N_ANGLE_STEPS, self.MAX_BINS, max_cells))
        for i in range(self.N_ANGLE_STEPS):
            for j in range(self.MAX_BINS):
                for k in range(np.shape(cell2grid_map.flat[new_map[i, j]])):
                    reduced_map[i, j, k] = cell2grid_map.flat[new_map[i, j]][k]
        print('Saving results')
        np.savez('OGrid_data/occ_map_new_{}_1601.npz'.format(int(factor)), bin2grid_map=reduced_map)


    def get_p(self):
        try:
            p = 1 - 1 / (1 + np.exp(self.grid))
        except RuntimeWarning:
            self.grid[np.nonzero(self.grid > 50)] = 50
            self.grid[np.nonzero(self.grid < -50)] = -50
            p = 1 - 1 / (1 + np.exp(self.grid))
            logger.debug('Overflow when calculating probability')
        return p

    # def get_binary_map(self):
    #     return (self.grid > self.binary_threshold).astype(np.float)

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
        new_data[np.nonzero(new_data < threshold)] = -self.o_zero
        new_data[np.nonzero(new_data > 0)] = 0.5 + self.o_zero
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


if __name__=="__main__":
    grid = OccupancyGrid(True, 0.7, 8)
    # grid.calc_map(2)
    # grid.calc_map(4)
    grid.calc_map(8)
    # grid.calc_map(16)
    # grid.calc_map(32)
    # print('Finished sucsessfully')
    # with np.load('OGrid_data/occ_map_2_1601.npz') as data:
    #     new_map = data['new_map']
    #     cell2grid_map = data['cell2grid_map']
    # a = 1