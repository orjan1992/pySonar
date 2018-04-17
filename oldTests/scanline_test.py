import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

thresh = 50
# scan = np.load('scanline2.npz')['scanline']
# scan[100:133] = 90
# smooth = np.convolve(scan, np.ones((10,))/10, mode='full')

smooth = np.array([0, 10, 10, 20, 10, 25, 25, 25, 30, 5, 5, 10])
# valley = extreme(smooth, np.less_equal, order=1)[0]
# peak = extreme(smooth, np.greater_equal, order=1)[0]
valley = signal.argrelextrema(smooth, np.less_equal, order=1)[0]
peak = signal.argrelextrema(smooth, np.greater_equal, order=1)[0]

# if peak[0] < valley[0]:
#     last_peak = -2
#     last_valley = -1
#     last_is_peak = False
# else:
#     last_peak = -1
#     last_valley = -2
#     last_is_peak = True

i_max = np.shape(peak)[0]
j_max = np.shape(valley)[0]
valley_mask = np.ones(j_max, dtype=bool)
peak_mask = np.ones(i_max, dtype=bool)

i = j = 1
# # TODO: Rememer first
while i < i_max and j < j_max:
    # Point in both
    if peak[i] == valley[j]:
        k = 1
        value = smooth[peak[i]]
        while i + k < i_max and j + k < j_max and (smooth[peak[i+k]] == value or smooth[valley[j+k]] == value):
            k += 1
        peak_mask[i:i+k] = False
        valley_mask[j:j+k] = False
        if smooth[peak[i]-1] < value and smooth[min(peak[i+k-1]+1, valley[j+k-1]+1)] > value:
            valley_mask[j + k // 2] = True
        if value > smooth[peak[i]-1] and value > smooth[min(peak[i+k-1]+1, valley[j+k-1]+1)]:
            peak_mask[i + k // 2] = True

        i += k
        j += k
        continue
    if valley[j - 1] < peak[i] < valley[j] or peak[i - 1] < valley[j] < peak[i]:
        i += 1
        j += 1
        continue
    if valley[j-1] < peak[i-1] <= peak[i] < valley[j]:
        peak_mask[i] = False
        i += 1
    if peak[i - 1] < valley[j - 1] <= valley[j] < peak[i]:
        valley_mask[j] = False
        j += 1
# i = 0
# i_max = len(peak)
# data_len = len(smooth)
# while i < i_max:
#     k = 1
#     while i + k < data_len and smooth[i] >= smooth[i+k]:
#         k +=1
#     if k > 1:
#         peak_mask[i:k-1] = False
#         peak_mask[k-1] = True



peak = peak[peak_mask]
valley = valley[valley_mask]


print(np.in1d(peak, valley))

# plt.plot(scan)
plt.plot(smooth)
plt.scatter(peak, smooth[peak], color='orange')
plt.scatter(valley, smooth[valley], color='green', marker='x')
# plt.scatter(peak[max_mask], smooth[peak[max_mask]], color='red')
# print(len(peak), len(valley))
# if peak[0] < valley[0]:
#     peak = peak[1:]
# if peak[-1] > valley[-1]:
#     peak = peak[:-1]
# print(len(peak), len(valley))
# mask = np.logical_or(smooth[peak] - smooth[valley[:-2]] > thresh, smooth[peak] - smooth[valley[2:]] > thresh)
# plt.scatter(peak[mask]*800/len(scan), smooth[peak[mask]], color='red')
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