import numpy as np
import cv2
from ast import literal_eval
import matplotlib.pyplot as plt

wp_list = literal_eval('[(801, 801), (926, 370), (947, 327), (964, 314), (985, 309), (1031, 320), (1178, 379), (1218, 390), (1235, 388), (1252, 380), (1281, 351), (1268, 367), (1292, 321), (1280, 271), (1167, 0)]')
im = np.zeros((801, 1601), dtype=np.uint8)
tmp = np.array(wp_list, dtype=np.int32).reshape((-1,1,2))
cv2.polylines(im, [tmp], False, (255, 0, 0), 600)
cv2.polylines(im, [tmp], False, (230, 0, 0), 400)
cv2.polylines(im, [tmp], False, (200, 0, 0), 200)
cv2.polylines(im, [tmp], False, (150, 0, 0), 100)
cv2.polylines(im, [tmp], False, (100, 0, 0), 50)
plt.imshow(im)
plt.show()