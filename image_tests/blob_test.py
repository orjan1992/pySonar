import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read image
im = np.load('test.npz')['olog'].astype(np.uint8)

im = cv2.blur(im, (3,3))
tmp = cv2.applyColorMap(im, cv2.COLORMAP_HOT)


# Blob detector
# params = cv2.SimpleBlobDetector_Params()
# params.minThreshold = 10
# params.maxThreshold = 255
# params.filterByArea = False
# params.filterByCircularity = False
# params.minCircularity = 0.1
# params.filterByConvexity = False
# params.minConvexity = 0.87
# params.filterByInertia = False
# params.minInertiaRatio = 0.01
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(im)
lim = 50

detector = cv2.FastFeatureDetector_create()
detector.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
keypoints = detector.detect(im)
L = len(keypoints)
x = np.zeros(L)
y = np.zeros(L)
counter = 0
map = np.zeros(L, dtype=np.uint8)
for i in range(0, L):
    x[i] = keypoints[i].pt[1]
    y[i] = keypoints[i].pt[0]
for i in range(0, L):
    x2 = np.power((x[i] - x), 2)
    y2 = np.power((y[i] - y), 2)
    r = np.sqrt(x2 + y2) < lim
    if map[i] != 0:
        map[r] = map[i]
    else:
        counter += 1
        map[r] = counter

labels = [[] for i in range(np.argmax(map))]

for i in range(0, L):
    labels[map[i]].append(keypoints[i])

im_with_keypoints = tmp
counter = 0
for keypoints in labels:
    if len(keypoints) > 1:
        R = np.random.randint(0, 255)
        G = np.random.randint(0, 255)
        B = np.random.randint(0, 255)
        im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (R, G, B),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(counter)
# im_with_keypoints = cv2.drawKeypoints(tmp, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# binary_descriptor = cv2.createLineSegmentDetector()
# lines = binary_descriptor.detect(im)[0]
# im_with_keypoints = binary_descriptor.drawSegments(np.zeros(np.shape(im), dtype=np.uint8), lines)
# im_with_keypoints = binary_descriptor.drawSegments(tmp, lines)


# thresh = 20
# canny = cv2.Canny(im, thresh, thresh*2, 3)
# im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# im_with_keypoints = cv2.drawContours(tmp, contours, -1, (255, 0, 0), 2)



cv2.imshow("keypoints", im_with_keypoints)
cv2.waitKey(0)