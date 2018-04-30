from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
from ogrid.occupancyGrid import *
from settings import *
import cv2

tmp = loadmat('obs.mat')
grid = tmp['grid']
range_scale = tmp['range_scale']


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
contours = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
contours_dilated = contours
convex_contour_list = []
for c in contours:
    convex_contour_list.append(cv2.convexHull(c, returnPoints=True))
contours = convex_contour_list

savemat('data.mat', {'orig': grid, 'thresh': thresh, 'contour_orig': contours1, 'contour_filter': contours2, 'contours_dilated': contours_dilated, 'contours_convex': convex_contour_list})
np.savez('data.npz', orig=grid, thresh=thresh, contour_orig=contours1, contour_filter=contours2, contours_dilated=contours_dilated, contours_convex=convex_contour_list)