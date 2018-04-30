from scipy.io import loadmat, savemat
from pySonarLog.save.voronoi_ex.collisionAvoidance_ex import CollisionAvoidance
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


tmp = np.load('data.npz')
contours = tmp['contours_convex']


co = CollisionAvoidance()
co.update_obstacles(contours, 30)
co.update_pos(msg())
co.update_external_wps([[0, 0, 0], [40, 0, 0]])
# co.bin_map = cv2.drawContours(np.zeros((GridSettings.height, GridSettings.width), dtype=np.uint8), contours, -1, (255, 255, 255), -1)
co.main_loop(True)

