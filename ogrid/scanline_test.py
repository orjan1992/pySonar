import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

thresh = 50
scan = np.load('scanline2.npz')['scanline']
scan[100:133] = 90
smooth = np.convolve(scan, np.ones((10,))/10, mode='full')
# min = extreme(smooth, np.less_equal, order=1)[0]
# max = extreme(smooth, np.greater_equal, order=1)[0]
min = signal.argrelextrema(smooth, np.less_equal, order=1)[0]
max = signal.argrelextrema(smooth, np.greater_equal, order=1)[0]

tmp_min = np.nonzero(np.in1d(min, max))[0]
if np.any(tmp_min):
    tmp_max = np.nonzero(np.in1d(max, min))[0]
    max_mask = np.ones(np.shape(max), dtype=bool)
    min_mask = np.ones(np.shape(min), dtype=bool)
    for i_min, i_max in zip(tmp_min, tmp_max):
        if i_min != 0 and i_max != 0:
            if min[i_min] > max[i_max]:
                min_mask[i_min] = False
            elif min[i_min] < max[i_max]:
                max_mask[i_max] = False
            else:
                min_mask[i_min] = False
                max_mask[i_max] = False
    min = min[min_mask]
    max = max[max_mask]

if max[0] < min[0]:
    max = max[1:]
if max[-1] > min[-1]:
    max = max[-1]

if len(min) - 2 != len(max):
    if len(min) - 2 > len(max):
        a = 1 # TODO: Samme som for else
    else:
        max_mask = np.ones(np.shape(max), dtype=bool)
        k = 0
        i = 0
        while i < len(max):  # TODO: Muligens ikke pluss her
            if max[i] < min[i - k]:
                max_mask[i] = False
                k += 1
            i += 1
        # max = max[max_mask]


max_mask = np.nonzero(np.logical_not(max_mask))
plt.plot(scan)
plt.plot(smooth)
plt.scatter(max, smooth[max], color='orange')
plt.scatter(min, smooth[min], color='green')
plt.scatter(max[max_mask], smooth[max[max_mask]], color='red')
# print(len(max), len(min))
# if max[0] < min[0]:
#     max = max[1:]
# if max[-1] > min[-1]:
#     max = max[:-1]
# print(len(max), len(min))
# mask = np.logical_or(smooth[max] - smooth[min[:-2]] > thresh, smooth[max] - smooth[min[2:]] > thresh)
# plt.scatter(max[mask]*800/len(scan), smooth[max[mask]], color='red')
plt.show()

# # maxtab, mintab = peakdet(smooth,20)
# plt.plot(scan)
# plt.plot(smooth)
# # plt.scatter(ext, smooth[ext], color='red')
# plt.plot(ext[0], smooth[ext], color='red')
#
# # plt.scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
# # plt.scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
# plt.show()