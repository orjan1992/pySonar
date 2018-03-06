from scipy.spatial import Voronoi
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra
from coordinate_transformations import sat2uint
from settings import GridSettings, CollisionSettings

class MyVoronoi(Voronoi):
    connection_matrix = 0

    def __init__(self, points):
        super(MyVoronoi, self).__init__(points)

    def add_wp_as_gen_point(self, point_index):
        if point_index < 0:
            region_index = self.point_region[np.shape(self.points)[0] + point_index]
        else:
            region_index = self.point_region[point_index]

        self.vertices = np.append(self.vertices, [self.points[point_index]], axis=0)
        new_vertice = np.shape(self.vertices)[0]-1

        for i in range(np.shape(self.regions[region_index])[0]):
            self.ridge_vertices.append([int(new_vertice), int(self.regions[region_index][i])])
        return new_vertice

    def add_wp(self, wp):
        dist = np.sqrt(np.square(self.points[:, 0] - wp[0]) + np.square(self.points[:, 1] - wp[1]))
        point_index = np.argmin(dist)
        region_index = self.point_region[point_index]

        self.vertices = np.append(self.vertices, [wp], axis=0)
        new_vertice = np.shape(self.vertices)[0] - 1
        new_ridges = []
        for i in range(np.shape(self.regions[region_index])[0]):
            self.ridge_vertices.append([int(new_vertice), int(self.regions[region_index][i])])
            new_ridges.append(self.ridge_vertices[-1])
        return new_vertice, new_ridges
            
    def gen_obs_free_connections(self, contours, shape):
        center = self.points.mean(axis=0)
        ptp_bound = self.points.ptp(axis=0)

        for i in range(np.shape(self.ridge_vertices)[0]):
            if self.ridge_vertices[i][0] == -1 or self.ridge_vertices[i][1] == -1:
                try:
                    if self.ridge_vertices[i][0] == -1:
                        j = self.ridge_vertices[i][1]
                    else:
                        j = self.ridge_vertices[i][0]

                    t = self.points[self.ridge_points[i][1]] - self.points[self.ridge_points[i][0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = self.points[self.ridge_points[i]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = self.vertices[j] + direction * ptp_bound.max()

                    self.vertices = np.append(self.vertices, [far_point], axis=0)
                    if self.ridge_vertices[i][0] == -1:
                        self.ridge_vertices[i][0] = np.shape(self.vertices)[0]-1
                    else:
                        self.ridge_vertices[i][1] = np.shape(self.vertices)[0]-1
                except IndexError:
                    break

        bin = cv2.drawContours(np.zeros(shape, dtype=np.uint8), contours, -1, (255, 255, 255), -1)

        self.connection_matrix = np.zeros((np.shape(self.vertices)[0], np.shape(self.vertices)[0]))

        for i in range(np.shape(self.ridge_vertices)[0]):
            p1x = sat2uint(self.vertices[self.ridge_vertices[i][0]][0], GridSettings.width)
            p1y = sat2uint(self.vertices[self.ridge_vertices[i][0]][1], GridSettings.height)
            p2x = sat2uint(self.vertices[self.ridge_vertices[i][1]][0], GridSettings.width)
            p2y = sat2uint(self.vertices[self.ridge_vertices[i][1]][1], GridSettings.height)
            lin = cv2.line(np.zeros(np.shape(bin), dtype=np.uint8), (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)

            if not np.any(np.logical_and(bin, lin)):
                # tmp = False
                if self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] == 0:
                    self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] = self.connection_matrix[
                        self.ridge_vertices[i][1], self.ridge_vertices[i][0]] = np.sqrt(
                        (self.vertices[self.ridge_vertices[i][1]][0] -
                         self.vertices[self.ridge_vertices[i][0]][0]) ** 2 +
                        (self.vertices[self.ridge_vertices[i][1]][1] -
                         self.vertices[self.ridge_vertices[i][0]][1]) ** 2)

    def add_start_penalty(self, ridges):
        """
        Add extra weight to ridges with large change in heading
        :param ridges: list of ridges
        :return:
        """
        # TODO: Penalty should probably be range dependent
        for ridge in ridges:
            penalty = np.abs((np.arctan2(self.vertices[ridge[0]][0] - self.vertices[ridge[1]][0],
                                         self.vertices[ridge[0]][1] - self.vertices[ridge[1]][1]) + np.pi / 2 % np.pi) -
                             np.pi / 2) * CollisionSettings.start_penalty_factor
            self.connection_matrix[ridge[0], ridge[1]] *= penalty
            self.connection_matrix[ridge[1], ridge[0]] *= penalty

    def dijkstra(self, start_in, stop_in):
        if start_in < 0:
            start = start_in + np.shape(self.vertices)[0]
        else:
            start = start_in
        if stop_in < 0:
            stop = stop_in + np.shape(self.vertices)[0]
        else:
            stop = stop_in
        # dist_matrix, predecessors = dijkstra(self.connection_matrix, indices=start,
        #                                      directed=False, return_predecessors=True)
        # length = 0
        predecessors = dijkstra(self.connection_matrix, indices=start,
                                             directed=False, return_predecessors=True)[1]
        shortest_path = []
        i = stop
        while i != start:
            if i == -9999:
                return None
            shortest_path.append(i)
            # length += dist_matrix[i]
            i = predecessors[i]
        shortest_path.append(start)
        shortest_path.reverse()
        # print(length)
        return shortest_path
