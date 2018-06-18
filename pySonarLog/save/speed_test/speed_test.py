from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
from ogrid.occupancyGrid import *
from settings import *
from collision_avoidance.collisionAvoidance import CollisionAvoidance
import cv2

# tmp = loadmat('obs.mat')
# grid = tmp['grid']
# range_scale = tmp['range_scale']
tmp = np.load('data.npz')
grid = tmp['orig']
range_scale = 30

p_log_threshold = np.log(GridSettings.p_binary_threshold / (1 - GridSettings.p_binary_threshold))

thresh = (grid > p_log_threshold).astype(dtype=np.uint8)

contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
contours1 = contours

# Removing small contours
min_area = FeatureExtraction.min_area * range_scale
new_contours = list()
for contour in contours:
    if cv2.contourArea(contour) > min_area:
        new_contours.append(contour)
im2 = cv2.drawContours(np.zeros(np.shape(grid), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)
contours2 = new_contours

# dilating to join close contours and use safety margin
k_size = int(np.round(3 * 801.0 / range_scale))
im3 = cv2.dilate(im2, np.ones((k_size, k_size), dtype=np.uint8), iterations=1)
# im3 = cv2.dilate(im2, kernel, iterations=FeatureExtraction.iterations)



# contour_list.append(cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1])
# contour_list.append(cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1])
# contour_list.append(cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1])
CollisionSettings.send_new_wps = False
Settings.save_paths = False
Settings.save_collision_info = False
Settings.save_scan_lines = False
Settings.save_obstacles = False

class msg:
    north = 0
    east = 0
    yaw = 0

    def to_tuple(self):
        return self.north, self.east, self.yaw
from timeit import default_timer as timer

t = []
t_total = []
t_mean = []
co = CollisionAvoidance(None, None)
co.update_pos(msg())
co.update_external_wps([[0, 0, 0], [40, 0, 0]])

t.append([])
i = 0
start = timer()
for j in range(10):
    print('j={}'.format(j))
    t2 = timer()
    contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
    contour_list = []
    for c in contours:
        contour_list.append(cv2.convexHull(c, returnPoints=True))
    co.update_obstacles(contour_list, 30)
    co.update_external_wps([[0, 0, 0], [40, 0, 0]])
    co.main_loop(True)
    t[i].append(timer()-t2)
t_total.append(timer()-start)
t_mean.append(np.mean(np.array(t[i])))

t.append([])
i = 1
start = timer()
for j in range(10):
    print('j={}'.format(j))
    t2 = timer()
    contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
    co.update_external_wps([[0, 0, 0], [40, 0, 0]])
    co.update_obstacles(contours, 30)
    co.main_loop(True)
    t[i].append(timer()-t2)
t_total.append(timer()-start)
t_mean.append(np.mean(np.array(t[i])))

t.append([])
i = 2
start = timer()
for j in range(10):
    print('j={}'.format(j))
    t2 = timer()
    contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    co.update_external_wps([[0, 0, 0], [40, 0, 0]], 30)
    co.update_obstacles(contours, range_scale)
    co.main_loop(True)
    t[i].append(timer()-t2)
t_total.append(timer()-start)
t_mean.append(np.mean(np.array(t[i])))

text_list = ['Convex', 'TC89', 'Simple']
print('Name\t\tMean\t\tTotal')
for i in range(len(text_list)):
    print('{}\t|\t{}\t|\t{}'.format(text_list[i], t_mean[i]*1000, t_total[i]))
