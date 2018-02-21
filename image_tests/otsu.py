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
    a, b = cv2.threshold(im, np.mean(im[im != 0]), 255, cv2.THRESH_TOZERO)
    plt.imshow(im)
    plt.show()
    # im = cv2.blur(im, (3,3))
    # tmp = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
    # ret, thr = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    # im2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(tmp, contours, -1, (0,255,0), 3)
    cv2.imshow("preview", im)
    cv2.waitKey(0)
