from ogrid.rawGrid import RawGrid
import numpy as np
from settings import *
import logging
logger = logging.getLogger('OccupancyGrid')

class OccupancyGrid(RawGrid):
    # counter = None
    # sign = None

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
        self.size = int((self.RES - 1) / cell_factor)
        self.occ_grid = np.full((self.size, self.size), self.p_log_zero, dtype=self.oLog_type)
        self.occ2raw_matrix = np.ones((cell_factor, cell_factor))
        try:
            with np.load('ogrid/OGrid_data/occ_map_{}_1601.npz'.format(int(cell_factor))) as data:
                self.angle2cell = data['angle2cell']
                self.angle2cell_rad = data['angle2cell_rad']
        except Exception as e:
            self.calc_map(cell_factor)

    def occ2raw(self):
        self.grid = np.kron(self.occ_grid, self.occ2raw_matrix)

    def raw2occ(self):
        for i in range(self.size):
            for j in range(self.size):
                self.occ_grid[i, j] = np.mean(self.grid[i:i+self.cell_factor, j:j+self.cell_factor])

    def calc_map(self, factor):
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
        contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
        im = cv2.applyColorMap(((self.grid + 6)*255.0 / 12.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        im = cv2.drawContours(im, contours, -1, (255, 0, 0), 2)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), contours

if __name__=="__main__":
    grid = OccupancyGrid(True, 0.3, 0.9, 0.7, 0.75, 16)
    grid.grid = np.random.random(np.shape(grid.grid))
    grid.raw2occ()

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(grid.occ_grid)
    plt.figure(2)
    plt.imshow(grid.grid)
    plt.show()

    # plt.plot(grid.rad_map)
    # plt.show()
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