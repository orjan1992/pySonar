from scipy.spatial import Voronoi

class MyVoronoi(Voronoi):
    def __init__(self, points):
        super().__init__(points)

    # def add_wp(self, index):
    #     for vertice in super.regions[index]:
    #         super.
    #         p2x = int(vp.vertices[vertice][0])
    #         p2y = int(vp.vertices[vertice][1])
    #         if p2x >= 0 and p2y >= 0:
    #             cv2.line(new_im, WP_end, (p2x, p2y), (0, 255, 0), 1)
