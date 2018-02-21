import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Read image
    im = np.load('test.npz')['olog'].astype(np.uint8)
    hist, _ = np.histogram(im.ravel(), 256)
    hist = hist[1:]
    grad = np.gradient(hist)
    i = np.argmax(np.abs(grad) < 20)
    print(i)
    print(hist[i])
    _, thresh = cv2.threshold(im, i, 255, cv2.THRESH_TOZERO)
    plt.imshow(thresh)
    plt.show()

    # plt.subplot(121)
    # plt.plot(range(255), grad)
    # plt.ylim((-10000, 10000))
    # plt.subplot(122)
    # plt.plot(range(255), hist)
    # plt.ylim((0, 25000))
    # plt.show()