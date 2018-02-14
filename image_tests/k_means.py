import cv2
import numpy as np
import matplotlib.pyplot as plt
im = np.load('test.npz')['olog'].astype(np.float32)
Z = im.reshape((-1, 1))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,  10,  1.0)
k = 8
ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

fig = list()
ax = list()
for i in range(k):
    res = np.zeros(np.shape(label), dtype=np.uint8)
    res[label == i] = 1
    fig.append(plt.figure())
    ax.append(fig[i].add_subplot(111))
    ax[i].imshow(res.reshape(np.shape(im)).astype(np.uint8))
plt.show()
