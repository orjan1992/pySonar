import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read image
im = np.load('test.npz')['olog'].astype(np.uint8)

im = cv2.blur(im, (3,3))
tmp = cv2.applyColorMap(im, cv2.COLORMAP_HOT)


# Blob detector
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 255
params.filterByArea = False
params.filterByCircularity = False
params.minCircularity = 0.1
params.filterByConvexity = False
params.minConvexity = 0.87
params.filterByInertia = False
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im)
im_with_keypoints = cv2.drawKeypoints(np.zeros(np.shape(im), dtype=np.uint8), keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("keypoints", im_with_keypoints)
cv2.waitKey(0)