import numpy as np
import math
import logging
import cv2
from settings import FeatureExtraction
import threading
from settings import GridSettings, CollisionSettings

logger = logging.getLogger('RawGrid')


class RawGrid(object):
    cur_step = 0
    PI2 = math.pi / 2
    oLog_type = np.uint8
    old_delta_x = 0
    old_delta_y = 0
    old_delta_yaw = 0
    MIN_ROT = 0.1 * math.pi / 180
    i_max = 1601
    j_max = 1601

    last_bearing = 0
    MAX_BINS = 800
    MAX_CELLS = 4
    N_ANGLE_STEPS = 6400
    STRIDE = 1604
    RES = 1601
    r_unit = np.zeros((RES, RES))
    theta = np.zeros((RES, RES))
    theta_grad = np.zeros((RES, RES), dtype=np.uint16)
    x_mesh_unit = np.zeros((RES, RES))
    y_mesh_unit = np.zeros((RES, RES))
    map = np.zeros((N_ANGLE_STEPS, MAX_BINS, MAX_CELLS), dtype=np.uint32)
    last_data = np.zeros(MAX_BINS, dtype=np.uint8)
    last_distance = None
    range_scale = 1.0
    last_dx = 0
    last_dy = 0
    reliable = False
    im = np.zeros((RES, RES), dtype=float)
    contours = []

    rot_remainder = 0
    dx_remainder = 0
    dy_remainder = 0

    def __init__(self, half_grid, p_zero=0):
        if half_grid:
            self.i_max = int((RawGrid.RES / 2) * (1 + math.tan(math.pi / 90.0)))
        self.origin_j = self.origin_i = np.round((RawGrid.RES - 1) / 2).astype(int)
        self.p_log_zero = p_zero
        self.grid = np.full((self.i_max, self.j_max), self.p_log_zero, dtype=self.oLog_type)
        [self.i_max, self.j_max] = np.shape(self.grid)
        if not np.any(RawGrid.r_unit != 0):
            try:
                with np.load('ogrid/OGrid_data/rad_1601.npz') as data:
                    RawGrid.x_mesh_unit = data['x_mesh']
                    RawGrid.y_mesh_unit = data['y_mesh']
                    RawGrid.r_unit = data['r']
                    RawGrid.theta = data['theta']
                    RawGrid.theta_grad = data['theta_grad']
            except Exception as e:
                xy_unit = np.linspace(-(RawGrid.RES - 1) / 2, (RawGrid.RES - 1) / 2, RawGrid.RES, True) / (0.5*RawGrid.RES)
                RawGrid.x_mesh_unit, RawGrid.y_mesh_unit = np.meshgrid(xy_unit, xy_unit)
                # RawGrid.r_unit = np.sqrt(np.power(RawGrid.x_mesh_unit, 2) + np.power(RawGrid.y_mesh_unit, 2))
                RawGrid.r_unit = np.sqrt(RawGrid.x_mesh_unit**2 + RawGrid.y_mesh_unit**2)
                RawGrid.theta = np.arctan2(RawGrid.y_mesh_unit, RawGrid.x_mesh_unit)
                RawGrid.theta_grad = RawGrid.theta * 3200.0 // np.pi
                np.savez('ogrid/OGrid_data/rad_1601.npz', x_mesh=RawGrid.x_mesh_unit,
                         y_mesh=RawGrid.y_mesh_unit, r=RawGrid.r_unit, theta=RawGrid.theta, theta_grad=RawGrid.theta_grad)

        if not np.any(RawGrid.map != 0):
            try:
                with np.load('ogrid/OGrid_data/map_1601.npz') as data:
                    RawGrid.map = data['map']
            except:
                raise RuntimeError('NO raw map file')
        self.lock = threading.Lock()



    def get_raw(self):
        return self.grid

    def update_raw(self, msg):
        if msg.bearing > 6399 or msg.bearing < 0:
            return
        self.range_scale = msg.range_scale
        range_step = self.MAX_BINS / msg.dbytes
        if GridSettings.scale_raw_data:
            new_data = np.zeros(self.MAX_BINS, dtype=np.int)
        else:
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

        if GridSettings.scale_raw_data:
            new_data = np.clip(((new_data - msg.ad_low)*255.0/msg.ad_span), 0, 255).astype(np.uint8)

        bearing_diff = msg.bearing - self.last_bearing
        beam_half = 27
        if msg.chan2:
            beam_half = 13

        self.lock.acquire()
        if math.fabs(bearing_diff) <= msg.step:
            if bearing_diff > 0:
                value_gain = (new_data.astype(float) - self.last_data) / bearing_diff
                for n in range(self.last_bearing, msg.bearing + 1):
                    for i in range(0, self.MAX_CELLS):
                        self.grid.flat[RawGrid.map[n, :, i]] = self.last_data + (n - self.last_bearing) * value_gain
                for n in range(msg.bearing + 1, msg.bearing + beam_half):
                    for i in range(0, self.MAX_CELLS):
                        self.grid.flat[RawGrid.map[n, :, i]] = new_data
            else:
                value_gain = (new_data.astype(float) - self.last_data) / (-bearing_diff)
                for n in range(msg.bearing, self.last_bearing + 1):
                    for i in range(0, self.MAX_CELLS):
                        self.grid.flat[RawGrid.map[n, :, i]] = self.last_data + (n - msg.bearing) * value_gain
                for n in range(msg.bearing - beam_half, msg.bearing):
                    for i in range(0, self.MAX_CELLS):
                        self.grid.flat[RawGrid.map[n, :, i]] = new_data
        else:
            for n in range(msg.bearing - beam_half, msg.bearing + beam_half):
                for i in range(0, self.MAX_CELLS):
                    self.grid.flat[RawGrid.map[n, :, i]] = new_data

        self.lock.release()

        self.last_bearing = msg.bearing
        self.last_data = new_data

    def update_distance(self, distance):
        # TODO: fix negative indexes
        # try:
        #     factor = distance / self.last_distance
        # except TypeError:
        #     factor = 1
        #     self.last_distance = distance
        # except ZeroDivisionError:
        #     factor = 1
        #     self.last_distance = distance
        # if factor == 1:
        #     return
        # if factor < 0:
        #     print('distance: {},\told_distance: {}'.format(distance, self.last_distance))
        # new_grid = np.full(np.shape(self.grid), self.p_log_zero, dtype=self.oLog_type)
        # try:
        #     if factor < 1:
        #         # old distance > new distance
        #         new_grid = self.grid[np.meshgrid((np.round((np.arange(0, self.j_max, 1) - self.origin_j) *
        #                                                    factor + self.origin_j)).astype(dtype=int),
        #                                          (np.round((np.arange(0, self.i_max, 1) - self.origin_i) *
        #                                                    factor + self.origin_i)).astype(dtype=int))]
        #     else:
        #         # old distance < new distance
        #         i_lim = int(round(0.5 * self.i_max / factor))
        #         j_lim = int(round(0.5 * self.j_max / factor))
        #         new_grid[i_lim:-i_lim, j_lim:-j_lim] = self.grid[
        #             np.meshgrid((np.round((np.arange(j_lim, self.j_max - j_lim, 1) - self.origin_j) *
        #                                   factor + self.origin_j)).astype(dtype=int),
        #                         (np.round((np.arange(i_lim, self.i_max - i_lim, 1) - self.origin_i) *
        #                                   factor + self.origin_i)).astype(dtype=int))]
        # except IndexError:
        #     logger.warning('Could not update grid size.')
        #     self.last_distance = distance
        #     return
        # self.lock.acquire()
        # self.grid = new_grid
        # self.lock.release()
        
        self.last_distance = distance

    def clear_grid(self):
        self.lock.acquire()
        self.grid = np.full((self.i_max, self.j_max), self.p_log_zero)
        self.lock.release()
        
        logger.info('Grid cleared')

    def calc_obstacles(self, threshold):
        # Finding histogram, calculating gradient
        hist = np.histogram(self.grid.astype(np.uint8).ravel(), 256)[0]
        # Check if more than half of pixels is set
        # self.reliable = hist[0] < GridSettings.max_unset_pixels
        # if not self.reliable:
        #     return cv2.cvtColor(cv2.applyColorMap(self.grid.astype(np.uint8), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB), None
        grad = np.gradient(hist[1:])
        i = np.argmax(np.abs(grad) > threshold)
        print(i)
        if i == 0:
            return cv2.cvtColor(cv2.applyColorMap(self.grid.astype(np.uint8), cv2.COLORMAP_HOT),
                                cv2.COLOR_BGR2RGB), None

        # threshold based on gradient
        thresh = cv2.threshold(self.grid.astype(np.uint8), i, 255, cv2.THRESH_BINARY)[1]
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

        # Removing small contours
        min_area = FeatureExtraction.min_area * self.range_scale
        new_contours = list()
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                new_contours.append(contour)
        im2 = cv2.drawContours(np.zeros(np.shape(self.grid), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)

        # dilating to join close contours
        k_size = np.round(CollisionSettings.obstacle_margin * 801.0 / self.range_scale).astype(int)
        im3 = cv2.dilate(im2, np.ones((k_size, k_size), dtype=np.uint8), iterations=1)
        _, contours, _ = cv2.findContours(im3, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        im = cv2.applyColorMap(self.grid.astype(np.uint8), cv2.COLORMAP_HOT)
        im = cv2.drawContours(im, contours, -1, (255, 0, 0), 1)
        self.im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.contours = contours

    def get_obstacles(self):
        """
        calculates obstacles
        :return: image, contours
        """

        return self.im, self.contours

    def rot(self, dyaw):
        dyaw += self.rot_remainder
        if abs(dyaw) < GridSettings.min_rot:
            self.rot_remainder = dyaw
            return False
        try:
            # new_grid = cv2.warpAffine(self.grid, cv2.getRotationMatrix2D((self.origin_i, self.origin_j), dyaw*180.0/np.pi, 1.0), (self.RES, self.RES), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, self.p_log_zero)
            new_grid = cv2.warpAffine(self.grid, cv2.getRotationMatrix2D((self.origin_i, self.origin_j), dyaw*180.0/np.pi, 1.0), (self.RES, self.RES), cv2.INTER_LINEAR, borderValue=self.p_log_zero)
            self.lock.acquire()
            self.grid = new_grid
            self.lock.release()
            self.rot_remainder = 0
        except TypeError:
            a = 1

        return True

    def trans(self, dx, dy):
        dx = dx * RawGrid.MAX_BINS / self.range_scale + self.last_dx
        dy = dy * RawGrid.MAX_BINS / self.range_scale + self.last_dy
        dx_int = np.round(dx).astype(int)
        dy_int = np.round(dy).astype(int)
        self.last_dx = dx - dx_int
        self.last_dy = dy - dy_int
        # logger.info((dx_int, dy_int))
        if dx_int < GridSettings.min_trans and dy_int < GridSettings.min_trans:
            self.last_dx += dx_int
            self.last_dy += dy_int
            return False
        self.lock.acquire()
        if dy_int == 0:
            if dx_int > 0:
                self.grid[dx_int:, :] = self.grid[:-dx_int, :]
                self.grid[:dx_int, :] = self.p_log_zero
            elif dx_int < 0:
                self.grid[:dx_int, :] = self.grid[-dx_int:, :]
                self.grid[dx_int:, :] = self.p_log_zero
        elif dy_int > 0:
            if dx_int > 0:
                self.grid[dx_int:, :-dy_int] = self.grid[:-dx_int, dy_int:]
                self.grid[:dx_int, :] = self.p_log_zero
                self.grid[:, -dy_int:] = self.p_log_zero
            elif dx_int < 0:
                self.grid[:dx_int, :-dy_int] = self.grid[-dx_int:, dy_int:]
                self.grid[dx_int:, :] = self.p_log_zero
                self.grid[:, -dy_int:] = self.p_log_zero
            else:
                self.grid[:, :-dy_int] = self.grid[:, dy_int:]
                self.grid[:, -dy_int:] = self.p_log_zero
        else:
            if dx_int > 0:
                self.grid[dx_int:, -dy_int:] = self.grid[:-dx_int, :dy_int]
                self.grid[:dx_int, :] = self.p_log_zero
                self.grid[:, :-dy_int] = self.p_log_zero
            elif dx_int < 0:
                self.grid[:dx_int, -dy_int:] = self.grid[-dx_int:, :dy_int]
                self.grid[dx_int:, :] = self.p_log_zero
                self.grid[:, :-dy_int] = self.p_log_zero
            else:
                self.grid[:, -dy_int:] = self.grid[:, :dy_int]
                self.grid[:, :-dy_int] = self.p_log_zero
        self.lock.release()

    def trans_and_rot(self, diff):
        return self.rot(diff.dyaw) and self.trans(diff.dx, diff.dy)


if __name__ == "__main__":
    from ogrid.occupancyGrid import OccupancyGrid
    import matplotlib.pyplot as plt
    grid = OccupancyGrid(False, GridSettings.p_inital, GridSettings.p_occ, GridSettings.p_free,
                         GridSettings.p_binary_threshold, 16)
    grid.last_distance = 50
    grid.randomize()
    tmp = grid.grid.copy()
    grid.update_distance(30)
    plt.subplot(121)
    plt.imshow(grid.grid)
    plt.subplot(122)
    plt.imshow(tmp)
    plt.show()

