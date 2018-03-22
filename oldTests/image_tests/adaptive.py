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
    _, thresh = cv2.threshold(im, i, 255, cv2.THRESH_TOZERO)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = list()
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            new_contours.append(contour)
    print('length countour: {}'.format(len(new_contours)))
    im2 = cv2.drawContours(np.zeros(np.shape(im), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)
    im3 = cv2.dilate(im2, np.ones((11, 11), dtype=np.uint8), iterations=1)
    im4, contours, hierarchy = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im5 = cv2.drawContours(im4, new_contours, -1, (255, 255, 255), -1)
    for contour in contours:
        ellipse = cv2.fitEllipse(contour)
        im5 = cv2.ellipse(im5, ellipse, (255, 0, 0), 1)

    cv2.imshow('test', im5)
    cv2.waitKey()