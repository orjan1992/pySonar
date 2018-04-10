import shapefile
import numpy as np
from collision_avoidance.voronoi import *
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore

class PathPlanner:
    def __init__(self, file):
        self.file = file
        sf = shapefile.Reader(file)
        self.shapes = sf.shapes()[:30]
        bbox = self.shapes[0].bbox
        for shape in self.shapes:
            if shape.bbox[0] < bbox[0]:
                bbox[0] = shape.bbox[0]
            if shape.bbox[1] < bbox[1]:
                bbox[1] = shape.bbox[1]
            if shape.bbox[2] > bbox[2]:
                bbox[2] = shape.bbox[2]
            if shape.bbox[3] > bbox[3]:
                bbox[3] = shape.bbox[3]
        self.offset = np.array(bbox[:2])
        self.range = np.array(bbox[2:]) - self.offset
        for shape in self.shapes:
            shape.points = np.array(shape.points) - self.offset
        # print(bbox)

    def find_new_wp(self, wp1, wp2):
        points = self.gen_point_list()
        vp = MyVoronoi(points)
        start_wp, start_ridges = vp.add_wp(wp1 - self.offset)
        end_wp, _ = vp.add_wp(wp2 - self.offset)
        vp.gen_obs_free_connections(self.gen_contour_list(), (np.round(self.range[1]).astype(np.int64), np.round(self.range[0]).astype(np.int64)))
        wps = vp.dijkstra(start_wp, end_wp)
        wp_list = []
        if wps is not None:
            for wp in wps:
                wp_list.append((int(vp.vertices[wp][0]), int(vp.vertices[wp][1])))
        return wp_list

    def gen_point_list(self):
        points =[]
        for shape in self.shapes:
            points.extend(np.round(shape.points).astype(np.int64))
        return np.array(points)

    def gen_contour_list(self):
        c_list = []
        for i in range(len(self.shapes)):
            print(i)
            vec = np.array(self.shapes[i].points)
            for j in range(len(self.shapes[i].parts) - 1):
                # contour = np.ndarray((self.shapes[i].parts[j + 1] - self.shapes[i].parts[j + 1], 1, 2))
                # contour[:, 0, :] = vec[self.shapes[i].parts[j]:self.shapes[i].parts[j + 1], :]
                c_list.append((np.round(vec[self.shapes[i].parts[j]:self.shapes[i].parts[j + 1], :].reshape((self.shapes[i].parts[j + 1] - self.shapes[i].parts[j], 1, 2)))).astype(np.int64))
        return c_list

    def plot(self, plot_widget):
        tmp = self.shapes
        for i in range(len(self.shapes)):
            print(i)
            vec = np.array(self.shapes[i].points)
            for j in range(len(self.shapes[i].parts) - 1):
                plot_widget.plot(vec[self.shapes[i].parts[j]:self.shapes[i].parts[j + 1], 0], vec[self.shapes[i].parts[j]:self.shapes[i].parts[j + 1], 1])
            # if len(shape.parts) > 1:
            #     plot_widget.plot(vec[shape.parts[-1]:, 0], vec[shape.parts[-1]:, 1])


if __name__ == '__main__':
    path_planner = PathPlanner('collision_avoidance/Mapfiles/Snorre B WGS84/Installation_line')
    # wp1 = (458000 - 100, 6821650 - 200)
    # wp2 = (458080, 6821780)
    wp1 = (994 + path_planner.offset[0].astype(np.int64), 933 + path_planner.offset[1].astype(np.int64))
    wp2 = (993 + path_planner.offset[0].astype(np.int64), 983 + path_planner.offset[1].astype(np.int64))
    qt = False
    if qt:
        app = QtGui.QApplication([])
        p1 = pg.plot()
        # vp = path_planner.find_new_wp(wp1, wp2, p1)

        p1.setXRange(0, path_planner.range[0])
        p1.setYRange(0, path_planner.range[1])
        path_planner.plot(p1)
        p1.show()

        app.exec_()
    else:
        from scipy.spatial import voronoi_plot_2d
        import matplotlib.pyplot as plt
        a = path_planner.gen_contour_list()
        import cv2
        im = cv2.drawContours(np.zeros((path_planner.range[0].astype(np.int64), path_planner.range[1].astype(np.int64), 3), dtype=np.uint8), a, -1, (255, 0, 0), 1)
        plt.imshow(im)
        plt.gca().invert_yaxis()
        plt.show()
        wps = path_planner.find_new_wp(wp1, wp2)
        for i in range(len(wps)-1):
            cv2.line(im, wps[i], wps[i+1], (0, 0, 255), 1)
        wp1_norm = (wp1[0] - path_planner.offset[0].astype(np.int64), wp1[1] - path_planner.offset[1].astype(np.int64))
        wp2_norm = (wp2[0] - path_planner.offset[0].astype(np.int64), wp2[1] - path_planner.offset[1].astype(np.int64))
        cv2.circle(im, wp1_norm, 50, (0, 255, 0))
        cv2.circle(im, wp2_norm, 100, (0, 255, 0))
        plt.imshow(im)
        plt.gca().invert_yaxis()
        plt.show()

        #
        b = 1
