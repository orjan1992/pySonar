from collision_avoidance.voronoi import MyVoronoi
import numpy as np
import cv2
from settings import *
from oldTests.collisionAvoidance_log import *

# Read image
im = np.load('test.npz')['olog'].astype(np.uint8)
# Finding histogram, calculating gradient
hist = np.histogram(im.ravel(), 256)[0][1:]
grad = np.gradient(hist)
i = np.argmax(np.abs(grad) < 10)

# threshold based on gradient
thresh = cv2.threshold(im, i, 255, cv2.THRESH_BINARY)[1]
_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Removing small contours
new_contours = list()
for contour in contours:
    if cv2.contourArea(contour) > FeatureExtraction.min_area:
        new_contours.append(contour)
im2 = cv2.drawContours(np.zeros(np.shape(im), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)

# dilating to join close contours
im3 = cv2.dilate(im2, FeatureExtraction.kernel, iterations=FeatureExtraction.iterations)
_, contours, _ = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

new_im = np.zeros((np.shape(im)[0], np.shape(im)[1], 3), dtype=np.uint8)
new_im = cv2.drawContours(new_im, contours, -1, (255, 255, 255), -1)

######
## init
######
CollisionSettings.send_new_wps = False
collision_avoidance = CollisionAvoidance(None)
collision_avoidance.update_pos(0, 0, 0)
# collision_avoidance.waypoint_list = [[0, 0, 1, 1], [10, 10, 2, 2], [15, 20, 3, 3]]
# collision_avoidance.waypoint_list = [[0, 0, 1, 1], [15, 20, 3, 3]]
# collision_avoidance.waypoint_list = [[0, 0, 1, 1], [29, -13, 3, 3]]
collision_avoidance.waypoint_list = [[0, 0, 1, 1], [10, -5, 3, 3], [30, -10, 3, 3]]
collision_avoidance.waypoint_counter = 1
collision_avoidance.update_obstacles(contours, 30)
collision_avoidance.check_collision_margins()

new_im = np.zeros((np.shape(collision_avoidance.bin_map)[0], np.shape(collision_avoidance.bin_map)[1], 3), np.uint8)
new_im[:, :, 0] = collision_avoidance.bin_map
new_im[:, :, 1] = collision_avoidance.bin_map
new_im[:, :, 2] = collision_avoidance.bin_map

WP0 = NED2grid(collision_avoidance.waypoint_list[0][0], collision_avoidance.waypoint_list[0][1], 0, 0, 0, 30)
cv2.circle(new_im, WP0, 2, (0, 0, 255), 2)
for i in range(len(collision_avoidance.waypoint_list) - 1):
    NE, constrained = constrainNED2range(collision_avoidance.waypoint_list[i + 1], collision_avoidance.waypoint_list[i], 0,
                                         0, 0, 30)
    if not constrained:
        WP1 = NED2grid(collision_avoidance.waypoint_list[i + 1][0], collision_avoidance.waypoint_list[i + 1][1], 0, 0, 0,
                       30)
        cv2.circle(new_im, WP1, 2, (255, 0, 0), 2)
        cv2.line(new_im, WP0, WP1, (255, 0, 0), 2)
        WP0 = WP1
    else:
        WP1 = NED2grid(NE[0], NE[1], 0, 0,
                       0, 30)
        cv2.circle(new_im, WP1, 2, (255, 0, 0), 2)
        cv2.line(new_im, WP0, WP1, (255, 0, 0), 2)
        break

new_im = np.zeros((np.shape(collision_avoidance.bin_map)[0], np.shape(collision_avoidance.bin_map)[1], 3), np.uint8)
new_im[:, :, 0] = collision_avoidance.bin_map
new_im[:, :, 1] = collision_avoidance.bin_map
new_im[:, :, 2] = collision_avoidance.bin_map
vp = collision_avoidance.calc_new_wp()
wp_list = collision_avoidance.new_wp_list
voronoi_wp_list = collision_avoidance.voronoi_wp_list
# # draw vertices
for ridge in vp.ridge_vertices:
    if ridge[0] != -1 and ridge[1] != -1:
        p1x = sat2uint(vp.vertices[ridge[0]][0], GridSettings.width)
        p1y = sat2uint(vp.vertices[ridge[0]][1], GridSettings.height)
        p2x = sat2uint(vp.vertices[ridge[1]][0], GridSettings.width)
        p2y = sat2uint(vp.vertices[ridge[1]][1], GridSettings.height)
        cv2.line(new_im, (p1x, p1y), (p2x, p2y), (0, 0, 255), 1)
for i in range(np.shape(vp.connection_matrix)[0]):
    for j in range(np.shape(vp.connection_matrix)[1]):
        if vp.connection_matrix[i, j] != 0:
            cv2.line(new_im, (sat2uint(vp.vertices[i][0], GridSettings.width),
                              sat2uint(vp.vertices[i][1], GridSettings.height)),
                     (sat2uint(vp.vertices[j][0], GridSettings.width)
                      , sat2uint(vp.vertices[j][1], GridSettings.height)), (0, 255, 0), 1)

collision_avoidance.calc_new_wp()
# draw route
WP0 = NED2grid(collision_avoidance.new_wp_list[0][0], collision_avoidance.new_wp_list[0][1], 0, 0, 0, 30)
cv2.circle(new_im, WP0, 2, (0, 0, 255), 2)
for i in range(len(collision_avoidance.new_wp_list)-1):
    NE, constrained = constrainNED2range(collision_avoidance.new_wp_list[i+1], collision_avoidance.new_wp_list[i], 0, 0, 0, 30)
    if not constrained:
        WP1 = NED2grid(collision_avoidance.new_wp_list[i+1][0], collision_avoidance.new_wp_list[i+1][1], 0, 0, 0, 30)
        cv2.circle(new_im, WP1, 2, (255, 0, 0), 2)
        cv2.line(new_im, WP0, WP1, (255, 0, 0), 2)
        WP0 = WP1
    else:
        WP1 = NED2grid(NE[0], NE[1], 0, 0,
                       0, 30)
        cv2.circle(new_im, WP1, 2, (255, 0, 0), 2)
        cv2.line(new_im, WP0, WP1, (255, 0, 0), 2)
        break

cv2.imshow('', new_im)
cv2.waitKey()
