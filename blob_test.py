import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read image
im = np.load('test.npz')['olog']
print(np.any(im > 255))
tmp = cv2.applyColorMap(im, cv2.COLORMAP_WINTER)
cv2.imshow("Keypoints", tmp)
cv2.waitKey(0)
# Set up the detector with default parameters.
detector = cv2.FastFeatureDetector_create()

# Detect blobs.
keypoints = detector.detect(im.astype(np.uint8))

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(tmp, keypoints, color=(0, 0, 255))
# Print all default params
print("Threshold: ", detector.getThreshold())
print("nonmaxSuppression: ", detector.getNonmaxSuppression())
print("neighborhood: ", detector.getType())
print("Total Keypoints with nonmaxSuppression: ", len(keypoints))
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)