from scipy.io import loadmat, savemat
# from pySonarLog.save.voronoi_ex.collisionAvoidance_ex import CollisionAvoidance
from collision_avoidance.collisionAvoidance import CollisionAvoidance
from settings import CollisionSettings
from settings import GridSettings
import numpy as np
import cv2
import logging
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class msg:
    north = 0
    east = 0
    yaw = 0

    def to_tuple(self):
        return self.north, self.east, self.yaw


tmp = np.load('data.npz')
contours = tmp['contours_convex']

CollisionSettings.send_new_wps = False
co = CollisionAvoidance()
co.update_obstacles(contours, 30)
co.update_pos(msg())
co.update_external_wps([[0, 0, 0], [40, 0, 0]])
co.main_loop(True)
# co2 = CollisionAvoidance()
# co2.update_obstacles(contours, 30)
# co2.update_pos(msg())
# co2.update_external_wps([[0, 0, 0], [40, 0, 0]])
# # co.bin_map = cv2.drawContours(np.zeros((GridSettings.height, GridSettings.width), dtype=np.uint8), contours, -1, (255, 255, 255), -1)
# for i in range(20):
#     co.main_loop(True)
#     co2.main_loop2(True)

