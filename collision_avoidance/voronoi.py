from scipy.spatial import Voronoi, ConvexHull
from matplotlib.path import Path
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra
from coordinate_transformations import sat2uint, constrainNED2range, NED2grid
from settings import GridSettings, CollisionSettings
import logging

logger = logging.getLogger('MyVoronoi')

class MyVoronoi(Voronoi):
    connection_matrix = 0
    shortest_path = []

    def __init__(self, points, *args, **kwargs):
        self.point_region = None
        self.regions = None
        self.vertices = None
        self.ridge_vertices = None
        super(MyVoronoi, self).__init__(points, *args, **kwargs)

    def add_wp(self, wp):
        region_index = self.point_region[np.argmin(np.sqrt(np.square(self.points[:, 0] - wp[0]) + np.square(self.points[:, 1] - wp[1])))]

        self.vertices = np.append(self.vertices, [wp], axis=0)
        new_vertice = np.shape(self.vertices)[0] - 1
        for i in range(np.shape(self.regions[region_index])[0]):
            self.ridge_vertices.append([int(new_vertice), int(self.regions[region_index][i])])
        return new_vertice, region_index
            
    def gen_obs_free_connections(self, range_scale, bin_map):
        """

        :param contours: list of contours
        :param shape: tuple, shape of grid
        :param add_penalty: add penalty to path away from old on
        :param old_wp_list: list of wp is grid frame
        :return:
        """

        self.connection_matrix = np.zeros((np.shape(self.vertices)[0], np.shape(self.vertices)[0]))

        line_width = np.round(CollisionSettings.vehicle_margin * 801 / range_scale).astype(int) # wp line width, considering vehicle size

        # Check if each ridge is ok, then calculate ridge
        # TODO: Calc dist with scipy.spatial.distance_matrix?
        for i in range(np.shape(self.ridge_vertices)[0]):
            # TODO: maybe error with ridge vertice == -1
            p1x = int(self.vertices[self.ridge_vertices[i][0]][0])
            p1y = int(self.vertices[self.ridge_vertices[i][0]][1])
            p2x = int(self.vertices[self.ridge_vertices[i][1]][0])
            p2y = int(self.vertices[self.ridge_vertices[i][1]][1])

            if (0 < p1x > GridSettings.width) and (0 < p2x > GridSettings.width) or\
                    (0 < p1y > GridSettings.width) and (0 < p2y > GridSettings.width):
                # Whole line is outside grid => no collision
                if self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] == 0:
                    self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] =\
                        self.connection_matrix[self.ridge_vertices[i][1], self.ridge_vertices[i][0]] = np.sqrt(
                        (self.vertices[self.ridge_vertices[i][1]][0] -
                         self.vertices[self.ridge_vertices[i][0]][0]) ** 2 +
                        (self.vertices[self.ridge_vertices[i][1]][1] -
                         self.vertices[self.ridge_vertices[i][0]][1]) ** 2)

            lin = cv2.line(np.zeros(np.shape(bin_map), dtype=np.uint8), (p1x, p1y), (p2x, p2y), (1, 0, 0), line_width)

            if not np.any(np.logical_and(bin_map, lin)):
                # No collision
                if self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] == 0:
                    self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] =\
                        self.connection_matrix[self.ridge_vertices[i][1], self.ridge_vertices[i][0]] = np.sqrt(
                        (self.vertices[self.ridge_vertices[i][1]][0] -
                         self.vertices[self.ridge_vertices[i][0]][0]) ** 2 +
                        (self.vertices[self.ridge_vertices[i][1]][1] -
                         self.vertices[self.ridge_vertices[i][0]][1]) ** 2)
            # else:
            #     if bin_map[p1y, p1x] == 0:


    def dijkstra(self, start_in, stop_in, collision_ind=None):
        if collision_ind is not None:
            self.connection_matrix[collision_ind, :] = 0
            self.connection_matrix[:, collision_ind] = 0
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
        length = 0
        dist, predecessors = dijkstra(self.connection_matrix, indices=start,
                                             directed=False, return_predecessors=True)
        self.shortest_path = []
        i = stop
        while i != start:
            if i == -9999:
                raise RuntimeError('No feasible path')
            self.shortest_path.append(i)
            length += dist[i]
            i = predecessors[i]
        self.shortest_path.append(start)
        self.shortest_path.reverse()
        logger.debug('path length={}'.format(length))
        return self.shortest_path
