import cv2
import numpy as np

# Read image
im = np.load('test.npz')['olog']
# Set up the detector with default parameters.
detector = cv2.FastFeatureDetector_create()

# Detect blobs.
keypoints = detector.detect(im.astype(np.uint8))

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im.astype(np.uint8), keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Print all default params
print("Threshold: ", detector.getThreshold())
print("nonmaxSuppression: ", detector.getNonmaxSuppression())
print("neighborhood: ", detector.getType())
print("Total Keypoints with nonmaxSuppression: ", len(keypoints))
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)