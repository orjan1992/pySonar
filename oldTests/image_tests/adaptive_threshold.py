import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Read image
    im = np.load('test.npz')['olog'].astype(np.uint8)
    # plt.hist(im.ravel(), 256, [0, 256])
    # plt.ylim(0, 10000)
    print(np.mean(im[im != 0]))
    im = cv2.GaussianBlur(im, (7,7), 0)

    th = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    cv2.imshow("preview", th)
    cv2.waitKey(0)
