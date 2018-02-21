import numpy as np
import math
import logging
import cv2
from settings import BlobDetectorSettings, FeatureExtraction
import threading

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
    last_dx = 0
    last_dy = 0

    def __init__(self, half_grid, p_m, binary_threshold=0.7, cellsize=0):
        if half_grid:
            self.i_max = int((OGrid.RES / 2) * (1 + math.tan(math.pi / 90.0)))
        self.cell_size = cellsize
        self.origin_j = self.origin_i = np.round((OGrid.RES - 1) / 2).astype(int)
        try:
            self.o_zero = math.log(p_m / (1 - p_m))
        except:
            self.o_zero = 0
        self.binary_threshold = math.log(binary_threshold / (1 - binary_threshold))
        self.o_log = np.ones((self.i_max, self.j_max), dtype=self.oLog_type) * self.o_zero
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

        self.lock = threading.Lock()
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
        new_data[np.nonzero(new_data < threshold)] = -self.o_zero
        new_data[np.nonzero(new_data > 0)] = 0.5 + self.o_zero
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
        except ZeroDivisionError:
            factor = 1
            self.last_distance = distance
        if factor == 1:
            return
        new_grid = np.ones(shape=np.shape(self.o_log), dtype=self.oLog_type) * self.o_zero
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
        
        self.o_log = np.ones((self.i_max, self.j_max)) * self.o_zero
        
        logger.info('Grid cleared')


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

    def get_obstacles_otsu(self):
        tmp = cv2.applyColorMap(self.o_log.astype(np.uint8), cv2.COLORMAP_HOT)
        ret, thr = cv2.threshold(self.o_log.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        im2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tmp, contours, -1, (0, 255, 0), 3)
        # return cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        return thr

    def adaptive_threshold(self, threshold):
        hist = np.histogram(self.o_log.astype(np.uint8).ravel(), 256)[0][1:]
        grad = np.gradient(hist)
        i = np.argmax(np.abs(grad) < threshold)
        print(i)
        thresh = cv2.threshold(self.o_log.astype(np.uint8), i, 255, cv2.THRESH_BINARY)[1]
        # return cv2.cvtColor(cv2.drawKeypoints(self.o_log.astype(np.uint8),
        #                                self.blob_detector.detect(thresh),
        #                                np.array([]), (0, 0, 255),
        #                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
        #              cv2.COLOR_BGR2RGB)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # TODO: Find all countours with small area and remove them
        # for contour in contours:
        #     if cv2.contourArea(contour) < FeatureExtraction.min_area:
        #         contours.remove(contour)
        im3 = cv2.dilate(im2, FeatureExtraction.kernel, iterations=FeatureExtraction.iterations)
        return im3


    def rot(self, dspi):

        dpsi_grad = dspi*3200/np.pi + self.old_delta_psi

        if abs(dpsi_grad) < 1:
            self.old_delta_psi = dpsi_grad
            return False
        else:
            self.old_delta_psi = dpsi_grad - np.round(dpsi_grad).astype(int)
        dpsi_grad = np.round(dpsi_grad).astype(int)
        new_grid = np.ones((self.i_max, self.j_max), dtype=self.oLog_type)*self.o_zero
        if dpsi_grad < 0:
            # new_grid.flat[OGrid.map[self.mask][1600-dpsi_grad:4801, :, :]] = self.o_log.flat[OGrid.map[self.mask][1600:4801+dpsi_grad, :, 0]]
            for n in range(1600-dpsi_grad, 4801):
                for i in range(0, self.MAX_CELLS):
                    new_grid.flat[OGrid.map[n, :, i]] = self.o_log.flat[OGrid.map[n+dpsi_grad, :, 0]]
        else:
            # new_grid.flat[OGrid.map[1600:4801-dpsi_grad, :, :]] = self.o_log.flat[OGrid.map[1600+dpsi_grad:4801, :, 0]]
            for n in range(1600, 4801-dpsi_grad):
                for i in range(0, self.MAX_CELLS):
                    new_grid.flat[OGrid.map[n, :, i]] = self.o_log.flat[OGrid.map[n+dpsi_grad, :, 0]]
        
        self.o_log = new_grid
        
        return True

    def trans(self, dx, dy):
        """
        :param dx: surge change
        :param dy: sway change
        :return:
        """
        # Transform to grid cordinates
        dx = dx*OGrid.MAX_BINS / self.range_scale + self.last_dx
        dy = dy*OGrid.MAX_BINS / self.range_scale + self.last_dy
        dx_int = np.round(dx).astype(int)
        dy_int = np.round(dy).astype(int)
        self.last_dx = dx - dx_int
        self.last_dy = dy - dy_int
        if dx_int == dy_int == 0:
            return False

        # move
        # new_grid = np.ones((self.i_max, self.j_max))*self.o_zero
        if dx_int > 0:
            if dy_int > 0:
                try:
                    self.o_log[dx_int:, :-dy_int] = self.o_log[:-dx_int, dy_int:]
                except Exception as e:
                    logger.debug('a\tself.o_log[{}:, :{}] = o_log[:{}, {}:]\n{}'.format(dx_int, -dy_int, -dx_int, dy_int, e))
                    return False
            elif dy_int < 0:
                try:
                    self.o_log[dx_int:, -dy_int:] = self.o_log[:-dx_int, :dy_int]
                except Exception as e:
                    logger.debug('b\tself.o_log[{}:, {}:] = o_log[:{}, :{}]\n{}'.format(dx_int, -dy_int, -dx_int, -dy_int, e))
                    return False
            else:
                try:
                    self.o_log[dx_int:, :] = self.o_log[:-dx_int, :]
                except Exception as e:
                    logger.debug('e\tself.o_log[{}:, :] = o_log[:{}, :]\n{}'.format(dx_int, -dx_int, e))
                    return False
        elif dx_int < 0:
            if dy_int > 0:
                try:
                    self.o_log[:-dx_int, :-dy_int] = self.o_log[dx_int:, dy_int:]
                except Exception as e:
                    logger.debug('c\tself.o_log[:{}, :{}] = o_log[{}:, {}:]\n{}'.format(-dx_int, -dy_int, dx_int, dy_int, e))
                    return False
            elif dy_int < 0:
                try:
                    self.o_log[:-dx_int, -dy_int:] = self.o_log[dx_int:, :dy_int]
                except Exception as e:
                    logger.debug('d\tself.o_log[:{}, {}:] = o_log[{}:, :{}]\n{}'.format(-dx_int, -dy_int, dx_int, dy_int, e))
                    return False
            else:
                try:
                    self.o_log[:-dx_int, :] = self.o_log[dx_int:, :]
                except Exception as e:
                    logger.debug('f\tself.o_log[:{}, :] = o_log[{}:, :]\n{}'.format(-dx_int, dx_int, e))
                    return False
        else:
            if dy_int > 0:
                try:
                    self.o_log[:, :-dy_int] = self.o_log[:, dy_int:]
                except Exception as e:
                    logger.debug('c\tself.o_log[:, :{}] = o_log[:, {}:]\n{}'.format(-dy_int, dy_int, e))
                    return False
            elif dy_int < 0:
                try:
                    self.o_log[:, -dy_int:] = self.o_log[:, :dy_int]
                except Exception as e:
                    logger.debug('d\tself.o_log[:, {}:] = o_log[:, :{}]\n{}'.format(-dy_int, dy_int, e))
                    return False
            else:
                return False
        # self.o_log = new_grid
        return True


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from messages.moosSonarMsg import *
    grid = OGrid(True, 0)
    grid.range_scale = 10
    grid.o_log = np.load('test.npz')['olog']
    grid.o_log[grid.o_log == grid.o_zero] = 0
    plt.subplot(211)
    plt.imshow(grid.o_log, vmin=0, vmax=255)

    with np.load('dxdy.npz') as data:
        dx = data['dx']
        dy = data['dy']

    grid.trans(dx, dy)

    plt.subplot(212)
    plt.imshow(grid.o_log, vmin=0, vmax=255)
    plt.show()

    # a = np.intersect1d(grid.map[2000, :, :], grid.map[2010, :, :])
    # print(len(a))
    # grid.o_log.flat[a] = 1
    # plt.imshow(grid.o_log)
    # plt.show()
