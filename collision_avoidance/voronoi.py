from scipy.spatial import Voronoi
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra

class MyVoronoi(Voronoi):
    def __init__(self, points):
        super(MyVoronoi, self).__init__(points)

    def add_wp(self, point_index):
        # TODO: Maybe wps should not be added as generation point?
        if point_index < 0:
            region_index = self.point_region[np.shape(self.points)[0] + point_index]
        else:
            region_index = self.point_region[point_index]

        self.vertices = np.append(self.vertices, [self.points[point_index]], axis=0)
        new_vertice = np.shape(self.vertices)[0]-1

        for i in range(np.shape(self.regions[region_index])[0]):
            self.ridge_vertices.append([int(new_vertice), int(self.regions[region_index][i])])
            
    def gen_obs_free_connections(self, contours, shape):
        bin = cv2.drawContours(np.zeros(shape, dtype=np.uint8), contours, -1, (255, 255, 255), -1)

        self.connection_matrix = np.zeros((np.shape(self.vertices)[0], np.shape(self.vertices)[0]))

        for i in range(np.shape(self.ridge_vertices)[0]):
            if self.ridge_vertices[i][0] > -1 and self.ridge_vertices[i][1] > -1:
                p1x = int(self.vertices[self.ridge_vertices[i][0]][0])
                p1y = int(self.vertices[self.ridge_vertices[i][0]][1])
                p2x = int(self.vertices[self.ridge_vertices[i][1]][0])
                p2y = int(self.vertices[self.ridge_vertices[i][1]][1])
                if p1x >= 0 and p2x >= 0 and p1y >= 0 and p2y >= 0:
                    lin = cv2.line(np.zeros(np.shape(bin), dtype=np.uint8), (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)
                    if not np.any(np.logical_and(bin, lin)):
                        if self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] == 0:
                            self.connection_matrix[self.ridge_vertices[i][0], self.ridge_vertices[i][1]] = self.connection_matrix[
                                self.ridge_vertices[i][1], self.ridge_vertices[i][0]] = np.sqrt(
                                (p2x - p1x) ** 2 + (p2y - p1y) ** 2)

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
        predecessors = dijkstra(self.connection_matrix, indices=start,
                                             directed=False, return_predecessors=True)[1]
        shortest_path = []
        i = stop
        while i != start:
            if i == -9999:
                return None
            shortest_path.append(i)
            i = predecessors[i]
        shortest_path.append(start)
        shortest_path.reverse()
        return shortest_path
