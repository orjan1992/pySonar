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

class msg(object):
    north = 0
    east = 0
    yaw = 0

    def __init__(self, pos):
        self.north = pos[0]
        self.east = pos[1]
        self.yaw = pos[2]

    def to_tuple(self):
        return self.north, self.east, self.yaw


tmp = loadmat('collision_info20180515-154041.mat')
contours = tmp['obstacles'][0]
range_scale = tmp['range_scale'][0]
pos = tmp['pos'][0]
old_wps = tmp['old_wps']
new_wps = tmp['new_wps']




CollisionSettings.send_new_wps = False
co = CollisionAvoidance()
co.update_obstacles(contours, range_scale)
co.update_pos(msg(pos))
co.update_external_wps(old_wps)
co.main_loop(True)
# co2 = CollisionAvoidance()
# co2.update_obstacles(contours, 30)
# co2.update_pos(msg())
# co2.update_external_wps([[0, 0, 0], [40, 0, 0]])
# # co.bin_map = cv2.drawContours(np.zeros((GridSettings.height, GridSettings.width), dtype=np.uint8), contours, -1, (255, 255, 255), -1)
# for i in range(20):
#     co.main_loop(True)
#     co2.main_loop2(True)